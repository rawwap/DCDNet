from model.DCDNet import DCDNet
from util.utils import count_params, set_seed, mIOU
import argparse
from copy import deepcopy
import os
import time
import torch
from torch.nn import CrossEntropyLoss, DataParallel
from torch.optim import SGD
from tqdm import tqdm
from data.dataset import FSSDataset
from prettytable import PrettyTable



def parse_args():
    parser = argparse.ArgumentParser(description='DCDNet for CD-FSS')
    parser.add_argument('--data-root',
                        type=str,
                        default='./dataset',
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
                        default=0.0005,
                        help='Learning rate for main model')
    parser.add_argument('--d_lr',
                        type=float,
                        default=0.0001,
                        help='Learning rate for discriminator')
    parser.add_argument('--refine', dest='refine', action='store_true', default=True)
    parser.add_argument('--shot',
                        type=int,
                        default=1,
                        help='Number of support image-mask pairs per episode')
    parser.add_argument('--cuda',
                        type=int,
                        default=0,
                        help='GPU device index')
    parser.add_argument('--seed',
                        type=int,
                        default=0,
                        help='Random seed for generating testing samples')
    parser.add_argument('--episode',
                        type=int,
                        default=48000,
                        total='Total number of training episodes')
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
                        default=True,
                        help='Enable fine-tuning mode')
    parser.add_argument('--model_train',
                        type=bool,
                        default=True,
                        help='Whether to train main model or only discriminator')
    return parser.parse_args()


def evaluate(model, dataloader, device, args):
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

def main():
    path_dir = 'ifa'

    args = parse_args()
    print('\n' + str(args))
    
    # cuda init
    os.environ['CUDA_VISIBLE_DEVICES'] = f"{args.cuda}"
    device = torch.device("cuda:{}".format(0))

    ### Please modify the following paths with your trained model paths.
    if args.dataset == 'deepglobe':
        if args.backbone == 'resnet50':
            if args.shot == 1:
                checkpoint_path = ''
            if args.shot == 5:
                checkpoint_path = ''
    if args.dataset == 'isic':
        if args.backbone == 'resnet50':
            if args.shot == 1:
                checkpoint_path = ''
            if args.shot == 5:
                checkpoint_path = ''
    if args.dataset == 'lung':
        if args.backbone == 'resnet50':
            if args.shot == 1:
                checkpoint_path = ''
            if args.shot == 5:
                checkpoint_path = ''
    if args.dataset == 'fss':
        if args.backbone == 'resnet50':
            if args.shot == 1:
                checkpoint_path = ''
            if args.shot == 5:
                checkpoint_path = ''
    
    miou = 0
    save_path = 'outdir/models/%s/%s' % (args.dataset, path_dir)
    os.makedirs(save_path, exist_ok=True)
 
    FSSDataset.initialize(img_size=400, datapath=args.data_root)
    train_dataset = args.dataset+'ifa'
    trainloader = FSSDataset.build_dataloader(train_dataset, args.batch_size, 4, '0', 'val', args.shot)
    FSSDataset.initialize(img_size=400, datapath=args.data_root)
    testloader = FSSDataset.build_dataloader(args.dataset, args.batch_size, 4, '0', 'test', args.shot)

    model = DCDNet(args)
    print('\nParams: %.1fM' % count_params(model))
    
    print('Loaded model:', checkpoint_path)

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint, False)
    

    for param in model.layer0.parameters():
        param.requires_grad = False
    for param in model.layer1.parameters():
        param.requires_grad = False

    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            for param in module.parameters():
                param.requires_grad = False

    criterion = CrossEntropyLoss(ignore_index=255)
    optimizer = SGD([param for param in model.parameters() if param.requires_grad],
                    lr=args.lr, momentum=0.9, weight_decay=5e-4)

    model = DataParallel(model).to(device)
    best_model = None

    iters = 0
    lr_decay_iters = [15, 25]
    model_stats_fp16 = calculate_model_size(model)

    previous_best = float(miou)
    # each snapshot is considered as an epoch
    for epoch in range(args.episode // args.snapshot):
        
        print("\n==> Epoch %i, learning rate = %.5f\t\t\t\t Previous best = %.2f"
              % (epoch, optimizer.param_groups[0]["lr"], previous_best))

        model.train()

        for module in model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.eval()

        total_loss = 0.0

        tbar = tqdm(trainloader)
        set_seed(int(time.time()))

        for i, (img_s_list, mask_s_list, img_q, mask_q, _, _, _) in enumerate(tbar):

            img_s_list = img_s_list.permute(1,0,2,3,4)
            mask_s_list = mask_s_list.permute(1,0,2,3)
            img_s_list = img_s_list.numpy().tolist()
            mask_s_list = mask_s_list.numpy().tolist()

            img_q, mask_q = img_q.to(device), mask_q.to(device)

            for k in range(len(img_s_list)):
                img_s_list[k], mask_s_list[k] = torch.Tensor(img_s_list[k]), torch.Tensor(mask_s_list[k])
                img_s_list[k], mask_s_list[k] = img_s_list[k].to(device), mask_s_list[k].to(device)

            output = model(img_s_list, mask_s_list, img_q, mask_q)
            
            mask_s = torch.cat(mask_s_list, dim=0)
            mask_s = mask_s.long()

            loss = criterion(output["Q_out"], mask_q) + criterion(output["out_refine"], mask_q) + criterion(output["self_out"], mask_q) + criterion(output["S_out"], mask_s) * 0.2 + \
                criterion(output["S_out_1"], mask_s) * 0.4 + criterion(output["Q_out_2"], mask_q) * 0.1 + criterion(output["S_out_2"], mask_s) * 0.1 + \
                criterion(output["Q_out_3"], mask_q) * 0.1 + criterion(output["S_out_3"], mask_s) * 0.1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            
            tbar.set_description('Loss: %.3f' % (total_loss / (i + 1)))

        iters += 1
        if iters in lr_decay_iters:
            optimizer.param_groups[0]['lr'] /= 10.0
        model.eval()
        set_seed(args.seed)
        miou = evaluate(model, testloader, device, args)

        if miou >= previous_best:
            best_model = deepcopy(model)
            previous_best = miou
            torch.save(best_model.module.state_dict(),
                os.path.join(save_path, '%s_%ishot_%.2f.pth' % (args.backbone, args.shot, miou)))
            
    print('\nEvaluating on 5 seeds.....')
    total_miou = 0.0
    for seed in range(5):
        print('\nRun %i:' % (seed + 1))
        set_seed(args.seed + seed)

        miou = evaluate(best_model, testloader,device , args)
        total_miou += miou

    print('\n' + '*' * 32)
    print('Averaged mIOU on 5 seeds: %.2f' % (total_miou / 5))
    print('*' * 32 + '\n')

    torch.save(best_model.module.state_dict(),
               os.path.join(save_path, '%s_%ishot_avg_%.2f.pth' % (args.backbone, args.shot, total_miou / 5)))

if __name__ == '__main__':
    main()
