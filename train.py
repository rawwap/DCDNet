import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from data.dataset import FSSDataset
from model.DCDNet import DCDNet
from util.utils import count_params, set_seed, mIOU
from tqdm import tqdm
from torch.nn import CrossEntropyLoss, DataParallel
from torch.optim import SGD
from copy import deepcopy
from collections import defaultdict
from model.discriminator import Discriminator
from datetime import datetime
from einops import repeat
import argparse
from prettytable import PrettyTable
from thop import profile


def parse_args():
    parser = argparse.ArgumentParser(description='DCDNet for CD-FSS')
    parser.add_argument('--data-root',
                        type=str,
                        default='../dataset',
                        help='Root path of training dataset')
    parser.add_argument('--dataset',
                        type=str,
                        default='fss',
                        choices=['fss', 'deepglobe', 'isic', 'lung'],
                        help='Training dataset name')
    parser.add_argument('--backbone',
                        type=str,
                        choices=['resnet50', 'resnet101'],
                        default='resnet50',
                        help='Backbone network for semantic segmentation model')
    parser.add_argument('--lr',
                        type=float,
                        default=0.001,
                        help='Learning rate for main model')
    parser.add_argument('--d_lr',
                        type=float,
                        default=0.0001,
                        help='Learning rate for discriminator')
    parser.add_argument('--refine', dest='refine', action='store_true', default=False,
                        help='Enable refinement module (default: False)')
    parser.add_argument('--shot',
                        type=int,
                        default=1,
                        help='Number of support image-mask pairs per episode')
    parser.add_argument('--cuda',
                        type=int,
                        default=0,
                        help='GPU device index (0-based)')
    parser.add_argument('--seed',
                        type=int,
                        default=0,
                        help='Random seed for generating testing samples')
    parser.add_argument('--episode',
                        type=int,
                        default=48000,
                        help='Total number of training episodes')
    parser.add_argument('--snapshot',
                        type=int,
                        default=1200,
                        help='Save model checkpoint after every N episodes')
    parser.add_argument('--batch-size',
                        type=int,
                        default=8,
                        help='Batch size for training')
    parser.add_argument('--nworker',
                        type=int,
                        default=8,
                        help='Number of workers for data loader')
    parser.add_argument('--latent_dim',
                        type=int,
                        default=1024,
                        help='Hidden dimension of discriminator')
    parser.add_argument('--units',
                        type=int,
                        default=256,
                        help='Number of hidden units in discriminator')
    parser.add_argument('--task_id',
                        type=int,
                        default=20,
                        help='Number of training classes')
    parser.add_argument('--lambda_',
                        type=float,
                        default=1.0,
                        help='Regularization coefficient for domain adaptation')
    parser.add_argument('--use_grl',
                        type=bool,
                        default=True,
                        help='Whether to use Gradient Reversal Layer in discriminator')
    parser.add_argument('--s_steps',
                        type=int,
                        default=1,
                        help='Training steps for main model per iteration')
    parser.add_argument('--d_steps',
                        type=int,
                        default=1,
                        help='Training steps for discriminator per iteration')
    parser.add_argument('--ce_loss_reg',
                        type=float,
                        default=1.0,
                        help='Weight coefficient for cross-entropy loss')
    parser.add_argument('--adv_loss_reg',
                        type=float,
                        default=0.005,
                        help='Weight coefficient for adversarial loss')
    parser.add_argument('--diff_loss_reg',
                        type=float,
                        default=0.1,
                        help='Weight coefficient for difference loss')
    parser.add_argument('--contr_loss_reg',
                        type=float,
                        default=0.05,
                        help='Weight coefficient for contrastive loss')
    parser.add_argument('--fintuning',
                        type=bool,
                        default=False,
                        help='Enable fine-tuning mode (default: False)')
    parser.add_argument('--model_train',
                        type=bool,
                        default=True,
                        help='Whether to train main model (True) or only discriminator (False)')
    return parser.parse_args()

def evaluate(model, dataloader, args):
    tbar = tqdm(dataloader)

    if args.dataset == 'fss':
        num_classes = 1000
    elif args.dataset == 'deepglobe':
        num_classes = 6
    elif args.dataset == 'isic':
        num_classes = 3
    elif args.dataset == 'lung':
        num_classes = 1

    metric = mIOU(num_classes)

    for i, (img_s_list, mask_s_list, img_q, mask_q, cls, _, id_q) in enumerate(tbar):

        img_s_list = img_s_list.permute(1,0,2,3,4)
        mask_s_list = mask_s_list.permute(1,0,2,3)

        img_s_list = img_s_list.numpy().tolist()
        mask_s_list = mask_s_list.numpy().tolist()

        img_q, mask_q = img_q.to(device), mask_q.to(device)

        for k in range(len(img_s_list)):
            img_s_list[k], mask_s_list[k] = torch.Tensor(img_s_list[k]), torch.Tensor(mask_s_list[k])
            img_s_list[k], mask_s_list[k] = img_s_list[k].to(device), mask_s_list[k].to(device)

        # https://github.com/niejiahao1998/IFA/issues/16
        # cls = cls + 1 # cls.shape: (b)
        # cls = repeat(cls, 'b -> b h w', h=mask_q.shape[1], w=mask_q.shape[2]).to(device) # cls: (b, h, w)
        # with torch.no_grad():
        #     output = model(img_s_list, mask_s_list, img_q, None)
        #     pred = torch.argmax(output["Q_out"], dim=1)

        # pred[pred == 1] = cls[pred == 1] # pred: (b, h, w)
        # mask_q[mask_q == 1] = cls[mask_q == 1].to(dtype = mask_q.dtype) # mask_q: (b, h, w)

        cls = cls[0].item()
        cls = cls + 1

        with torch.no_grad():
            output = model(img_s_list, mask_s_list, img_q, mask_q)
            pred = torch.argmax(output["Q_out"], dim=1)

        pred[pred == 1] = cls
        mask_q[mask_q == 1] = cls

        metric.add_batch(pred.cpu().numpy(), mask_q.cpu().numpy())

        tbar.set_description("Testing mIOU: %.2f" % (metric.evaluate() * 100.0))

    return metric.evaluate() * 100.0

def evaluate_discriminator(model, discriminator, dataloader, args):
    discriminator.eval()
    model.eval()

    total_correct = 0
    total_samples = 0
    class_correct = defaultdict(int)
    class_total = defaultdict(int)

    with torch.no_grad():
        tbar = tqdm(dataloader, desc='Evaluating Discriminator')
        for i, (img_s_list, mask_s_list, img_q, mask_q, cls, _, _) in enumerate(tbar):
            img_s_list = img_s_list.permute(1, 0, 2, 3, 4)
            mask_s_list = mask_s_list.permute(1, 0, 2, 3)

            img_s_list = [img.to(device) for img in img_s_list]
            mask_s_list = [mask.to(device) for mask in mask_s_list]
            img_q, mask_q = img_q.to(device), mask_q.to(device)
            cls_labels = cls.to(device)

            output = model(img_s_list, mask_s_list, img_q, mask_q)

            s_shared = output["S_shared"]

            predictions = discriminator(s_shared)
            _, preds = torch.max(predictions, 1)

            correct = (preds == cls_labels).sum().item()
            total_correct += correct
            total_samples += cls_labels.size(0)

            for label, pred in zip(cls_labels.cpu().numpy(), preds.cpu().numpy()):
                class_correct[label] += (pred == label)
                class_total[label] += 1

            tbar.set_postfix({
                'Overall Acc': f'{total_correct/total_samples:.2%}',
                'Current Acc': f'{correct/cls_labels.size(0):.2%}'
            })

    class_acc = {c: class_correct[c]/count for c, count in class_total.items()}
    avg_class_acc = sum(class_acc.values())/len(class_acc) if class_acc else 0

    return total_correct / total_samples


class DiffLoss(torch.nn.Module):
    # From: Domain Separation Networks (https://arxiv.org/abs/1608.06019)
    # Konstantinos Bousmalis, George Trigeorgis, Nathan Silberman, Dilip Krishnan, Dumitru Erhan

    def __init__(self):
        super(DiffLoss, self).__init__()

    def forward(self, D1, D2):
        D1=D1.view(D1.size(0), -1)
        D1_norm=torch.norm(D1, p=2, dim=1, keepdim=True).detach()
        D1_norm=D1.div(D1_norm.expand_as(D1) + 1e-6)

        D2=D2.view(D2.size(0), -1)
        D2_norm=torch.norm(D2, p=2, dim=1, keepdim=True).detach()
        D2_norm=D2.div(D2_norm.expand_as(D2) + 1e-6)

        return torch.mean((D1_norm.mm(D2_norm.t()).pow(2)))


class PrivateContraLoss(nn.Module):
    def __init__(self, num_classes=20, feature_dim=1024, temperature=0.07, momentum=0.99, init_scale=0.05):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.temperature = temperature
        self.momentum = momentum

        self.gmp = nn.AdaptiveMaxPool2d((1, 1))

        self.register_buffer("prototypes", init_scale * torch.randn(num_classes, feature_dim))
        self.register_buffer("init_counts", torch.zeros(num_classes))

        self.eps = 1e-8
        self.min_count = 3

    def forward(self, features, labels):

        pooled = self.gmp(features).view(features.size(0), -1)
        valid_mask = (labels < self.num_classes) & (labels >= 0)
        valid_labels = labels[valid_mask]
        valid_features = pooled[valid_mask]

        if valid_labels.size(0) > 0:
            self._update_prototypes(valid_features, valid_labels)
        
        if valid_labels.size(0) == 0:
            return torch.tensor(0., device=features.device)
        
        return self._contrastive_loss(valid_features, valid_labels)

    def _update_prototypes(self, features, labels):
        one_hot = F.one_hot(labels, self.num_classes).float()

        sum_features = torch.mm(one_hot.T, features)
        counts = one_hot.sum(dim=0)
        
        self.init_counts += counts
        
        with torch.no_grad():
            current_protos = sum_features / (counts[:, None] + self.eps)
            
            update_mask = self.init_counts >= self.min_count
            init_mask = ~update_mask
            
            self.prototypes[init_mask] = current_protos[init_mask]
            
            if update_mask.any():
                self.prototypes[update_mask] = (
                    self.momentum * self.prototypes[update_mask] +
                    (1 - self.momentum) * current_protos[update_mask]
                )

    def _contrastive_loss(self, features, labels):
        feat_norm = F.normalize(features, p=2, dim=1)
        proto_norm = F.normalize(self.prototypes, p=2, dim=1)
        
        logits = torch.mm(feat_norm, proto_norm.T) / self.temperature
        
        smoothed_labels = F.one_hot(labels, self.num_classes)
        smoothed_labels = smoothed_labels * 0.9 + 0.1 / self.num_classes
        
        return -(smoothed_labels * F.log_softmax(logits, dim=1)).sum(dim=1).mean()

def get_discriminator(args):
    discriminator=Discriminator(args).to(device)
    return discriminator

    
if __name__ == '__main__':  
    
    # args init
    args = parse_args()
    print('\n' + str(args))
    
    save_path = f'outdir/models/{args.dataset}/{time.strftime("%Y%m%d_%H%M%S")}'
    os.makedirs(save_path, exist_ok=True)
    
    # args logging
    log_file = os.path.join(save_path, 'training.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logging.info("Training Configuration:")
    for arg in vars(args):
        logging.info(f"{arg:20}: {getattr(args, arg)}")
    start_time = datetime.now()
    logging.info(f"\nTraining Start: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # device init
    os.environ['CUDA_VISIBLE_DEVICES'] = f"{args.cuda}"
    device = torch.device("cuda:{}".format(0))
    
    # dataset init
    FSSDataset.initialize(img_size=400, datapath=args.data_root)
    trainloader = FSSDataset.build_dataloader('pascal', args.batch_size, args.nworker, 4, 'trn', args.shot)
    testpascalloader = FSSDataset.build_dataloader('pascal', args.batch_size, args.nworker, 0, 'val', args.shot)
    testloader = FSSDataset.build_dataloader(benchmark=args.dataset, bsz=args.batch_size,
                                             nworker=args.nworker, fold='0', split='val', shot=args.shot)

    # model init
    model = DCDNet(args).to(device)
    print('\nParams: %.1fM' % count_params(model))
    discriminator = get_discriminator(args)
    model_checkpoint_path = './pretrained/Ori_SSP_trained_on_VOC.pth'
    print('Loaded discriminator:', model_checkpoint_path)
    model_checkpoint = torch.load(model_checkpoint_path)
    model.load_state_dict(model_checkpoint, strict=False)
    
    # main model training layer frozen
    for param in model.layer0.parameters():
        param.requires_grad = False
    for param in model.layer1.parameters():
        param.requires_grad = False
    for param in model.layer2.parameters():
        param.requires_grad = False
    for param in model.layer3.parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
       print(name, param.requires_grad)
    for module in model.modules():
       if isinstance(module, torch.nn.BatchNorm2d):
           for param in module.parameters():
               param.requires_grad = False
    
    # loss fuction & optimizer
    criterion = CrossEntropyLoss(ignore_index=255).to(device)
    adversarial_loss = CrossEntropyLoss().to(device)
    dis_adversarial_loss = CrossEntropyLoss().to(device)
    diffLoss = DiffLoss().to(device)
    contrastive_loss = PrivateContraLoss().to(device)
    optimizer = SGD([param for param in model.parameters() if param.requires_grad],
                    lr=args.lr, momentum=0.9, weight_decay=5e-4)
    d_optimizer = torch.optim.Adam([param for param in discriminator.parameters() if param.requires_grad], lr=args.d_lr, weight_decay=0.01)
    

    # training init
    iters = 0
    total_iters = args.episode // args.snapshot
    lr_decay_iters = [total_iters // 3, total_iters * 2 // 3]
    previous_best = 0
    best_acc = 0
    best_model = None
    
    
    # training
    for epoch in range(args.episode // args.snapshot):
        print(f"\n==> Epoch {epoch}, learning rate = {optimizer.param_groups[0]['lr']:.5f}\t\t Previous best = {previous_best:.2f}")
        epoch_start = time.time()
        logging.info(f"\n==> Epoch {epoch} started")
        
        model.train()
        for module in model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.eval()

        total_loss = 0.0
        total_ce_loss = 0.0
        loss_real = 0.0
        loss_fake = 0.0
        tbar = tqdm(trainloader)
        set_seed(int(time.time()))


        for i, (img_s_list, mask_s_list, img_q, mask_q, class_sample, _, _) in enumerate(tbar):
            # images ready
            img_s_list = img_s_list.permute(1,0,2,3,4)
            mask_s_list = mask_s_list.permute(1,0,2,3)
            img_s_list = img_s_list.numpy().tolist()
            mask_s_list = mask_s_list.numpy().tolist()

            img_q, mask_q = img_q.to(device), mask_q.to(device)
            for k in range(len(img_s_list)):
                img_s_list[k], mask_s_list[k] = torch.Tensor(img_s_list[k]), torch.Tensor(mask_s_list[k])
                img_s_list[k], mask_s_list[k] = img_s_list[k].to(device), mask_s_list[k].to(device)
            
            # main model training steps
            for _ in range(args.s_steps):
                output = model(img_s_list, mask_s_list, img_q, mask_q)
                mask_s = torch.cat(mask_s_list, dim=0).long()
                
                # task segmentation loss
                ce_loss = (
                    criterion(output["Q_out"], mask_q) +
                    criterion(output["self_out"], mask_q) +
                    0.2 * criterion(output["S_out"], mask_s) + 
                    0.4 * criterion(output["S_out_1"], mask_s)
                )
                
                # support & query share adversarial loss
                dis_s_out_training = discriminator(output["S_shared"])
                s_adv_loss = adversarial_loss(dis_s_out_training, class_sample.to(device))
                dis_q_out_training = discriminator(output["Q_shared"])
                q_adv_loss = adversarial_loss(dis_q_out_training, class_sample.to(device))
                adv_loss = s_adv_loss + q_adv_loss
                
                # share & private contrastive loss
                diff_loss = diffLoss(output["S_shared"], output["S_private"])
                
                # private contrastive loss
                contr_loss = contrastive_loss(output['S_private'], class_sample.to(device))
                    
                # total loss
                loss = args.ce_loss_reg * ce_loss + args.adv_loss_reg * adv_loss + args.diff_loss_reg * diff_loss + args.contr_loss_reg * contr_loss
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                 
                total_loss += loss.item()
                total_ce_loss += ce_loss.item()
            
            # discriminator training steps
            for _ in range(args.d_steps):
                
                if args.model_train==False:
                    output = model(img_s_list, mask_s_list, img_q, mask_q)
                
                d_optimizer.zero_grad()
                discriminator.zero_grad()
                
                # real data classification loss
                dis_q_real_out = discriminator(output['S_shared'].detach())
                dis_q_real_loss = dis_adversarial_loss(dis_q_real_out, class_sample.to(device))

                dis_real_loss = dis_q_real_loss
                dis_real_loss.backward(retain_graph=True)
                loss_real += dis_real_loss.item()
                
                # fake data classification loss
                z_fake = torch.randn(output['S_shared'].shape, dtype=torch.float32, device=device)
                dis_fake_out = discriminator(z_fake)
                dis_fake_loss = dis_adversarial_loss(dis_fake_out, torch.zeros_like(class_sample).to(device))
                dis_fake_loss.backward()
                loss_fake += dis_fake_loss.item()

                d_optimizer.step()
                

            tbar.set_description('Loss: %.3f | closs:%.3f | rloss: %.3f | floss: %.3f'
                                 % (total_loss / ((i + 1)*args.s_steps),(total_ce_loss / ((i + 1)*args.s_steps)),
                                    loss_real / ((i + 1)*args.d_steps), loss_fake / ((i + 1)*args.d_steps)))
            
        # evaluting model
        model.eval()
        discriminator.eval()
        set_seed(args.seed)
        miou = evaluate(model, testloader, args)
        acc = evaluate_discriminator(model, discriminator, testpascalloader, args)
        
        epoch_time = time.time() - epoch_start
        logging.info(f"Epoch {epoch} Summary:")
        logging.info(f"Time Cost:      {epoch_time:.2f}s")
        logging.info(f"Total Loss:     {total_loss / ((i + 1)*args.s_steps):.3f}")
        logging.info(f"mIOU:           {miou:.2f}")
        logging.info(f"acc:           {acc:.3f}")

        # model save
        if  miou >= previous_best and args.model_train == True:
            logging.info(f"New best model! mIOU improved to {miou:.2f}%")
            best_discriminator = deepcopy(discriminator)
            best_model = deepcopy(model)
            previous_best = miou
            best_acc = acc
            torch.save(best_model.state_dict(),
                       os.path.join(save_path, f'{args.backbone}_{args.shot}shot_{miou:.2f}.pth'))
            # torch.save(best_discriminator.state_dict(),
            #            os.path.join(save_path, f'discriminator_{args.shot}shot_{acc:.2f}_{miou:.2f}.pth'))
        
        # test discriminator model save
        if acc >= best_acc and args.model_train == False:
            best_discriminator = deepcopy(discriminator)
            best_model = deepcopy(model)
            best_acc = acc
            torch.save(best_model.state_dict(),
                       os.path.join(save_path, f'{args.backbone}_{args.shot}shot_{epoch}.pth'))
            # torch.save(best_discriminator.state_dict(),
            #            os.path.join(save_path, f'discriminator_{args.shot}shot_{acc:.2f}.pth'))
            
        iters += 1
        # main learn rate decay
        if iters in lr_decay_iters:
            optimizer.param_groups[0]['lr'] /= 10.0
        

    # testing model
    print('\nEvaluating on 5 seeds.....')
    logging.info("\nFinal Evaluation:")
    total_miou = 0.0
    for seed in range(5):
        print('\nRun %i:' % (seed + 1))
        set_seed(args.seed + seed)
        miou = evaluate(best_model, testloader, args)
        logging.info(f"Seed {seed+1}: mIOU = {miou:.2f}%")
        total_miou += miou
    print(f'\nAveraged mIOU on 5 seeds: {total_miou / 5:.2f}')
    end_time = datetime.now()
    training_duration = end_time - start_time
    logging.info("\nTraining Complete!")
    logging.info(f"Total Duration: {training_duration}")
    logging.info(f"Average mIOU:   {total_miou / 5:.2f}%")
    torch.save(best_model.state_dict(),
               os.path.join(save_path, f'{args.backbone}_{args.shot}shot_avg_{total_miou / 5:.2f}.pth'))
    # torch.save(best_discriminator.state_dict(),
    #            os.path.join(save_path, f'discriminator_{args.shot}shot_avg_{total_miou / 5:.2f}.pth'))
