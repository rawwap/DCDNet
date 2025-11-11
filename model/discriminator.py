import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, args):
        
        super(Discriminator, self).__init__()
        self.grl = GradientReversal(args.lambda_) if args.use_grl else nn.Identity()
        
        # conv
        self.conv_block = nn.Sequential(
            nn.Conv2d(args.latent_dim, 512, kernel_size=3, stride=2, padding=1),  # [B, 512, 25, 25]
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 256, kernel_size=3, stride=2, padding=1),   # [B, 256, 13, 13]
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 128, kernel_size=3, stride=2, padding=1),   # [B, 128, 7, 7]
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d((1, 1))                               # [B, 128, 1, 1]
        )
        self.fc_block = nn.Sequential(
            nn.Flatten(),                                             # [B, 128]
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, args.task_id)                               # [B, num_classes]
        )

    def forward(self, x):
        x = self.grl(x)
        x = self.conv_block(x)  
        x = self.fc_block(x)  
        return x

    def pretty_print(self, num):

        magnitude = 0
        while abs(num) >= 1000:
            magnitude += 1
            num /= 1000.0
        return '%.1f%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])

    def get_size(self):
        
        count = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print('Num parameters in Discriminator = %s ' % (self.pretty_print(count)))


class GradientReversalFunction(torch.autograd.Function):
    """
    From:
    https://github.com/jvanvugt/pytorch-domain-adaptation/blob/cb65581f20b71ff9883dd2435b2275a1fd4b90df/utils.py#L26

    Gradient Reversal Layer from:
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)
    Forward pass is the identity function. In the backward pass,
    the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_) 
        dx = -lambda_ * grads
        return dx, None


class GradientReversal(torch.nn.Module):
    def __init__(self, lambda_):
        super(GradientReversal, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)