import model.resnet as resnet

import torch
from torch import nn
import torch.nn.functional as F
import pdb

import numpy as np

class Modulation(nn.Module):
    def __init__(self, in_features=1024):
        super().__init__()
        self.interaction = nn.Sequential(
            nn.Conv2d(in_features*2, in_features//8, 3, padding=1),
            nn.GroupNorm(32, in_features//8),
            nn.ReLU(inplace=True)
        )
        self.param_generator = nn.Sequential(
            nn.Conv2d(in_features//8, in_features*2, 1),
            nn.Tanh()
        )
        
        nn.init.normal_(self.param_generator[0].weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.param_generator[0].bias)

    def forward(self, private, shared):
        
        combined = torch.cat([private, shared], dim=1)
        interaction = self.interaction(combined)
        
        params = self.param_generator(interaction)
        gamma, beta = torch.chunk(params, 2, dim=1)
        # stabilize early training by limiting modulation magnitude
        gamma = 0.1 * gamma
        beta = 0.1 * beta
        modulated = private * (1 + gamma) + beta
        return modulated

class DCDNet(nn.Module):
    def __init__(self, args):
        super(DCDNet, self).__init__()
        backbone = resnet.__dict__[args.backbone](pretrained=True)
        self.layer0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1, self.layer2, self.layer3 = backbone.layer1, backbone.layer2, backbone.layer3
        self.shared = Shared()
        self.private = Private()
        self.modulator = Modulation()
        self.fusion = Fusion()
        self.refine = args.refine
        self.fintuning = args.fintuning
        self.shot = args.shot
        self.iter_refine = False
        
        
    def forward(self, img_s_list, mask_s_list, img_q, mask_q):
        h, w = img_q.shape[-2:]

        output = {}
        feature_s_list = []
        
        # feature maps of support images
        for k in range(len(img_s_list)):
            with torch.no_grad():
                s_0 = self.layer0(img_s_list[k]) # [B, 128, 100, 100]
                s_1 = self.layer1(s_0) # [B, 256, 100, 100]
            s_2 = self.layer2(s_1) # [B, 512, 50, 50]
            s_3 = self.layer3(s_2) # [B, 1024, 50, 50]
            output['S_feature_init'] = s_3
            output['S_shared'] = self.shared(s_1)                     
            if self.fintuning == False:
                output['S_private'] = self.private(s_3)
            else:
                output['S_private'] = self.modulator(self.private(s_3), output['S_shared'])
            feature_s = self.fusion(output['S_feature_init'], output['S_shared'], output['S_private'], self.fintuning)
            feature_s_list.append(feature_s)
            del s_0,s_1,s_2,s_3
            
        feature_s_ls = torch.cat(feature_s_list, dim=0)
        output['S_feature'] = feature_s_ls
        
        # feature map of query image
        with torch.no_grad():
            q_0 = self.layer0(img_q)
            q_1 = self.layer1(q_0)
        q_2 = self.layer2(q_1)
        q_3 = self.layer3(q_2)
        output['Q_feature_init'] = q_3
        output['Q_shared'] = self.shared(q_1)   
        output['Q_private'] = self.private(q_3)
        if self.fintuning == False:
            output['Q_private'] = self.private(q_3)
        else: 
            output['Q_private'] = self.modulator(self.private(q_3), output['Q_shared'])
        del q_0,q_1,q_2,q_3
        feature_q = self.fusion(output['Q_feature_init'], output['Q_shared'], output['Q_private'], self.fintuning)
        output['Q_feature'] = feature_q
            
        feature_fg_list = []
        feature_bg_list = []
        supp_out_ls = []
        
        # support foreground & backgound feature
        for k in range(len(img_s_list)):
            feature_fg = self.masked_average_pooling(feature_s_list[k],
                                                               (mask_s_list[k] == 1).float())[None, :]
            feature_bg = self.masked_average_pooling(feature_s_list[k],
                                                               (mask_s_list[k] == 0).float())[None, :]
            feature_fg_list.append(feature_fg)
            feature_bg_list.append(feature_bg)
            
            if self.training:
                supp_similarity_fg = F.cosine_similarity(feature_s_list[k], feature_fg.squeeze(0)[..., None, None], dim=1)
                supp_similarity_bg = F.cosine_similarity(feature_s_list[k], feature_bg.squeeze(0)[..., None, None], dim=1)
                supp_out = torch.cat((supp_similarity_bg[:, None, ...], supp_similarity_fg[:, None, ...]), dim=1) * 10.0

                supp_out = F.interpolate(supp_out, size=(h, w), mode="bilinear", align_corners=True)
                supp_out_ls.append(supp_out)
        
        # average foreground prototypes & background prototypes
        FP = torch.mean(torch.cat(feature_fg_list, dim=0), dim=0).unsqueeze(-1).unsqueeze(-1)
        BP = torch.mean(torch.cat(feature_bg_list, dim=0), dim=0).unsqueeze(-1).unsqueeze(-1)
        
        if self.training:
            if self.fintuning:
                out_refine, out_1, supp_out_1, new_FP, new_BP = self.iter_BFP(FP, BP, feature_s_ls, feature_q, self.refine)
                out_1 = F.interpolate(out_1, size=(h, w), mode="bilinear", align_corners=True)
                supp_out_1 = F.interpolate(supp_out_1, size=(h, w), mode="bilinear", align_corners=True)
                ## iter = 2
                out_2, supp_out_2, new_FP, new_BP = self.iter_BFP(new_FP, new_BP, feature_s_ls, feature_q, self.iter_refine)
                out_2 = F.interpolate(out_2, size=(h, w), mode="bilinear", align_corners=True)
                supp_out_2 = F.interpolate(supp_out_2, size=(h, w), mode="bilinear", align_corners=True)
                ### iter = 3
                out_3, supp_out_3, new_FP, new_BP = self.iter_BFP(new_FP, new_BP, feature_s_ls, feature_q, self.iter_refine)
                out_3 = F.interpolate(out_3, size=(h, w), mode="bilinear", align_corners=True)
                supp_out_3 = F.interpolate(supp_out_3, size=(h, w), mode="bilinear", align_corners=True)
            else:
                if self.refine:
                    out_refine, out_1, supp_out_1, new_FP, new_BP = self.iter_BFP(FP, BP, feature_s_ls, feature_q, self.refine)
                else:
                    out_1, supp_out_1, new_FP, new_BP = self.iter_BFP(FP, BP, feature_s_ls, feature_q, self.refine)
                out_1 = F.interpolate(out_1, size=(h, w), mode="bilinear", align_corners=True)
                supp_out_1 = F.interpolate(supp_out_1, size=(h, w), mode="bilinear", align_corners=True)
        else:
            if self.refine:
                out_refine, out_1 = self.iter_BFP(FP, BP, feature_s_ls, feature_q, self.refine)
            else:
                out_1 = self.iter_BFP(FP, BP, feature_s_ls, feature_q, self.refine)
            
            out_1 = F.interpolate(out_1, size=(h, w), mode="bilinear", align_corners=True)

        if self.refine:
            out_refine = F.interpolate(out_refine, size=(h, w), mode="bilinear", align_corners=True)
            output["out_refine"] = out_refine
            output["Q_out"] = out_1
        else:
            output["Q_out"] = out_1
        
        if self.training:
            fg_q = self.masked_average_pooling(feature_q, (mask_q == 1).float())[None, :].squeeze(0)
            bg_q = self.masked_average_pooling(feature_q, (mask_q == 0).float())[None, :].squeeze(0)

            self_similarity_fg = F.cosine_similarity(feature_q, fg_q[..., None, None], dim=1)
            self_similarity_bg = F.cosine_similarity(feature_q, bg_q[..., None, None], dim=1)
            self_out = torch.cat((self_similarity_bg[:, None, ...], self_similarity_fg[:, None, ...]), dim=1) * 10.0

            self_out = F.interpolate(self_out, size=(h, w), mode="bilinear", align_corners=True)
            supp_out = torch.cat(supp_out_ls, 0)
            
            output["self_out"] = self_out
            output["S_out"] = supp_out
            output["S_out_1"] = supp_out_1
            
            if self.fintuning:
                # iter = 2
                output["Q_out_2"] = out_2
                output["S_out_2"] = supp_out_2
                # iter = 3
                output["Q_out_3"] = out_3
                output["S_out_3"] = supp_out_3
            
        return output
    
    # SSP function
    def SSP_func(self, feature_q, out):

        bs = feature_q.shape[0]
        pred_1 = out.softmax(1)
        pred_1 = pred_1.view(bs, 2, -1)
        pred_fg = pred_1[:, 1]
        pred_bg = pred_1[:, 0]
        fg_ls = []
        bg_ls = []
        fg_local_ls = []
        bg_local_ls = []
        for epi in range(bs):
            fg_thres = 0.7
            bg_thres = 0.6
            cur_feat = feature_q[epi].view(1024, -1)
            f_h, f_w = feature_q[epi].shape[-2:]
            if (pred_fg[epi] > fg_thres).sum() > 0:
                fg_feat = cur_feat[:, (pred_fg[epi]>fg_thres)] #.mean(-1)
            else:
                fg_feat = cur_feat[:, torch.topk(pred_fg[epi], 12).indices] #.mean(-1)
            if (pred_bg[epi] > bg_thres).sum() > 0:
                bg_feat = cur_feat[:, (pred_bg[epi]>bg_thres)] #.mean(-1)
            else:
                bg_feat = cur_feat[:, torch.topk(pred_bg[epi], 12).indices] #.mean(-1)
            # global proto
            fg_proto = fg_feat.mean(-1)
            bg_proto = bg_feat.mean(-1)
            fg_ls.append(fg_proto.unsqueeze(0))
            bg_ls.append(bg_proto.unsqueeze(0))

            # local proto
            fg_feat_norm = fg_feat / torch.norm(fg_feat, 2, 0, True) # 1024, N1
            bg_feat_norm = bg_feat / torch.norm(bg_feat, 2, 0, True) # 1024, N2
            cur_feat_norm = cur_feat / torch.norm(cur_feat, 2, 0, True) # 1024, N3

            cur_feat_norm_t = cur_feat_norm.t() # N3, 1024
            fg_sim = torch.matmul(cur_feat_norm_t, fg_feat_norm) * 2.0 # N3, N1
            bg_sim = torch.matmul(cur_feat_norm_t, bg_feat_norm) * 2.0 # N3, N2

            fg_sim = fg_sim.softmax(-1)
            bg_sim = bg_sim.softmax(-1)

            fg_proto_local = torch.matmul(fg_sim, fg_feat.t()) # N3, 1024
            bg_proto_local = torch.matmul(bg_sim, bg_feat.t()) # N3, 1024

            fg_proto_local = fg_proto_local.t().view(1024, f_h, f_w).unsqueeze(0) # 1024, N3
            bg_proto_local = bg_proto_local.t().view(1024, f_h, f_w).unsqueeze(0) # 1024, N3

            fg_local_ls.append(fg_proto_local)
            bg_local_ls.append(bg_proto_local)

        # global proto
        new_fg = torch.cat(fg_ls, 0).unsqueeze(-1).unsqueeze(-1)
        new_bg = torch.cat(bg_ls, 0).unsqueeze(-1).unsqueeze(-1)

        # local proto
        new_fg_local = torch.cat(fg_local_ls, 0).unsqueeze(-1).unsqueeze(-1)
        new_bg_local = torch.cat(bg_local_ls, 0)

        return new_fg, new_bg, new_fg_local, new_bg_local

    # COS similarity function
    def similarity_func(self, feature_q, fg_proto, bg_proto):
        similarity_fg = F.cosine_similarity(feature_q, fg_proto, dim=1)
        similarity_bg = F.cosine_similarity(feature_q, bg_proto, dim=1)

        out = torch.cat((similarity_bg[:, None, ...], similarity_fg[:, None, ...]), dim=1) * 10.0
        return out

    # MAP function
    def masked_average_pooling(self, feature, mask):
        mask = F.interpolate(mask.unsqueeze(1), size=feature.shape[-2:], mode='bilinear', align_corners=True)
        masked_feature = torch.sum(feature * mask, dim=(2, 3)) \
                         / (mask.sum(dim=(2, 3)) + 1e-5)
        return masked_feature

    # BFP function
    def iter_BFP(self, FP, BP, feature_s_ls, feature_q, refine=True):
        # input FP and BP are support prototype
        # SSP on query side
        # find the most similar part in query feature
        out_0 = self.similarity_func(feature_q, FP, BP)
        # SSP in query feature
        SSFP_1, SSBP_1, ASFP_1, ASBP_1 = self.SSP_func(feature_q, out_0)
        # update prototype for query prediction
        FP_1 = FP * 0.5 + SSFP_1 * 0.5
        BP_1 = SSBP_1 * 0.3 + ASBP_1 * 0.7
        # use updated prototype to search target in query feature
        out_1 = self.similarity_func(feature_q, FP_1, BP_1)
        # Refine (only for the 1st iter)
        if refine:
            # use updated prototype to find the most similar part in query feature again
            SSFP_2, SSBP_2, ASFP_2, ASBP_2 = self.SSP_func(feature_q, out_1)
            # update prototype again for query regine
            FP_2 = FP * 0.5 + SSFP_2 * 0.5
            BP_2 = SSBP_2 * 0.3 + ASBP_2 * 0.7
            FP_2 = FP * 0.5 + FP_1 * 0.2 + FP_2 * 0.3
            BP_2 = BP * 0.5 + BP_1 * 0.2 + BP_2 * 0.3
            # use updated prototype to search target in query feature again
            out_refine = self.similarity_func(feature_q, FP_2, BP_2)
            out_refine = out_refine * 0.7 + out_1 * 0.3

        # SSP on support side
        if self.training:
            # duplicate query prototype for support SSP if shot > 1
            if self.shot > 1:
                FP_nshot = FP.repeat_interleave(self.shot, dim=0)
                FP_1 = FP_1.repeat_interleave(self.shot, dim=0)
                BP_1 = BP_1.repeat_interleave(self.shot, dim=0)
            # find the most similar part in support feature list
            supp_out_0 = self.similarity_func(feature_s_ls, FP_1, BP_1)
            # SSP in support feature list
            SSFP_supp, SSBP_supp, ASFP_supp, ASBP_supp = self.SSP_func(feature_s_ls, supp_out_0)
            # update prototype for support prediction
            if self.shot > 1:
                FP_supp = FP_nshot * 0.5 + SSFP_supp * 0.5
            else:
                FP_supp = FP * 0.5 + SSFP_supp * 0.5

            BP_supp = SSBP_supp * 0.3 + ASBP_supp * 0.7
            # use updated prototype to search target in support feature list
            supp_out_1 = self.similarity_func(feature_s_ls, FP_supp, BP_supp)

            # process prototype if shot > 1
            if self.shot > 1:
                for i in range(FP_supp.shape[0]//self.shot):
                    for j in range(self.shot):
                        if j == 0:
                            FP_supp_avg = FP_supp[i*self.shot+j]
                            BP_supp_avg = BP_supp[i*self.shot+j]
                        else:
                            FP_supp_avg = FP_supp_avg + FP_supp[i*self.shot+j]
                            BP_supp_avg = BP_supp_avg + BP_supp[i*self.shot+j]

                    FP_supp_avg = FP_supp_avg/self.shot
                    BP_supp_avg = BP_supp_avg/self.shot
                    FP_supp_avg = FP_supp_avg.reshape(1,FP_supp.shape[1],FP_supp.shape[2],FP_supp.shape[3])
                    BP_supp_avg = BP_supp_avg.reshape(1,BP_supp.shape[1],BP_supp.shape[2],BP_supp.shape[3])
                    if i == 0:
                        new_FP_supp = FP_supp_avg
                        new_BP_supp = BP_supp_avg
                    else:
                        new_FP_supp = torch.cat((new_FP_supp,FP_supp_avg), dim=0)
                        new_BP_supp = torch.cat((new_BP_supp,BP_supp_avg), dim=0)

                FP_supp = new_FP_supp
                BP_supp = new_BP_supp          

        if refine:
            if self.training:
                return out_refine, out_1, supp_out_1, FP_supp, BP_supp
            else:
                return out_refine, out_1
        else:
            if self.training:
                return out_1, supp_out_1, FP_supp, BP_supp
            else:
                return out_1

# Shared layer
class Shared(nn.Module):
    def __init__(self, in_channels=256, hidden_channels=[512, 1024], 
                 kernel_sizes=[3, 3], dropout_rates=[0.2, 0.5]):
        super(Shared, self).__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels[0], 
                      kernel_size=kernel_sizes[0], padding=1),
            nn.GroupNorm(32, hidden_channels[0]),  
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rates[0])         
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(hidden_channels[0], hidden_channels[1], 
                      kernel_size=kernel_sizes[1], padding=1),
            nn.GroupNorm(32, hidden_channels[1]), 
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rates[1]),
            nn.MaxPool2d(2)                         
        )
        self.adaptor = nn.Sequential(
            nn.Conv2d(hidden_channels[1], hidden_channels[1], 1),
            nn.GroupNorm(32, hidden_channels[1]),
            nn.Sigmoid()  
        )
    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        attn = self.adaptor(x)
        return x * attn

# Private layer
class Private(nn.Module):
    def __init__(self, in_channels=1024, hidden_channels=[512, 1024], 
                 kernel_sizes=[3, 3], dropout_rates=[0.2, 0.5]):
        super(Private, self).__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels[0], 
                      kernel_size=kernel_sizes[0], padding=1),
            nn.GroupNorm(32, hidden_channels[0]),  
            nn.GELU(),                          
            nn.Dropout2d(dropout_rates[0])
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(hidden_channels[0], hidden_channels[1], 
                      kernel_size=kernel_sizes[1], padding=1),
            nn.GroupNorm(32, hidden_channels[1]),
            nn.GELU(),
            nn.Dropout2d(dropout_rates[1])
        )
        self.domain_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_channels[1], hidden_channels[1]//8, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels[1]//8, hidden_channels[1], 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        identity = x
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        domain_attn = self.domain_attn(x)
        return x * domain_attn + identity
    
# Fusion module
class Fusion(nn.Module):
    def __init__(self, feat_dim=1024, reduction=16):
        super().__init__()
        self.temperature = 0.7
        self.min_weight = 0.05
        self.channel_reduce = nn.Sequential(
            nn.Conv2d(feat_dim*3, feat_dim//reduction, 1),
            nn.GroupNorm(32, feat_dim//reduction),
            nn.ReLU(inplace=True)
        )
        self.weight_generator = nn.Sequential(
            nn.Conv2d(feat_dim//reduction, 3, kernel_size=3, padding=1),
            # logits; we'll apply temperature-softmax manually in forward
        )
        self.enhance = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim, 1),
            nn.GroupNorm(32, feat_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, base_feat, share_feat, private_feat, finetuning):
        
        combined = torch.cat([base_feat, share_feat, private_feat], dim=1)
        reduced = self.channel_reduce(combined)

        logits = self.weight_generator(reduced)  # [B,3,H,W]
        weights = F.softmax(logits / self.temperature, dim=1)
        # avoid collapse by flooring and renormalizing
        weights = torch.clamp(weights, min=self.min_weight)
        weights = weights / weights.sum(dim=1, keepdim=True)
        w_base = weights[:, 0:1, :, :]
        w_share = weights[:, 1:2, :, :]
        w_private = weights[:, 2:3, :, :] 

        fused = w_base * base_feat + w_share * share_feat + w_private * private_feat
        
        return self.enhance(fused) + fused