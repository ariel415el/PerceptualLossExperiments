import torch
from torch.nn import functional as F
from torch.nn.functional import conv2d


class GradLoss(torch.nn.Module):
    def __init__(self, batch_reduction='mean'):
        super(GradLoss, self).__init__()
        # self.ky = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).reshape(1, 1, 3, 3)
        # self.kx = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32).reshape(1, 1, 3, 3)
        self.ky = torch.tensor([[1, -1]], dtype=torch.float32).reshape(1, 1, 2, 1)
        self.kx = torch.tensor([[1], [-1]], dtype=torch.float32).reshape(1, 1, 1, 2)
        self.batch_reduction = batch_reduction
        self.name = 'GradLoss'

    def forward(self, im1, im2):

        self.kx = self.kx.to(im1.device)
        self.ky = self.ky.to(im1.device)
        im1_gray = torch.mean(im1, 1, keepdim=True)
        im2_gray = torch.mean(im2, 1, keepdim=True)
        diff_x = (conv2d(im1_gray, self.kx) - conv2d(im2_gray, self.kx)).abs()
        diff_y = (conv2d(im1_gray, self.ky) - conv2d(im2_gray, self.ky)).abs()

        diff_x = torch.mean(diff_x, dim=(1, 2, 3))
        diff_y = torch.mean(diff_y, dim=(1, 2, 3))
        loss = (diff_x + diff_y) / 2
        if self.batch_reduction == 'mean':
            loss = loss.mean()
        return loss



class GradLoss3Channels(torch.nn.Module):
    def __init__(self, batch_reduction='mean'):
        super(GradLoss3Channels, self).__init__()
        # self.ky = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).reshape(1, 1, 3, 3).repeat(1,3,1,1)
        # self.kx = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32).reshape(1, 1, 3, 3).repeat(1,3,1,1)
        self.ky = torch.tensor([[1, -1]], dtype=torch.float32).reshape(1, 1, 2, 1).repeat(3,1,1,1)
        self.kx = torch.tensor([[1], [-1]], dtype=torch.float32).reshape(1, 1, 1, 2).repeat(3,1,1,1)
        self.batch_reduction = batch_reduction
        self.name = 'GradLoss3ch'

    def forward(self, im1, im2):
        from torch.nn.functional import conv2d

        self.kx = self.kx.to(im1.device)
        self.ky = self.ky.to(im1.device)

        diff_x = (conv2d(im1, self.kx, groups=3) - conv2d(im2, self.kx, groups=3)).abs()
        diff_y = (conv2d(im1, self.ky, groups=3) - conv2d(im2, self.ky, groups=3)).abs()

        diff_x = torch.mean(diff_x, dim=(1, 2, 3))
        diff_y = torch.mean(diff_y, dim=(1, 2, 3))
        loss = (diff_x + diff_y) / 2
        if self.batch_reduction == 'mean':
            loss = loss.mean()
        return loss



class FastGradLoss(torch.nn.Module):
    def __init__(self, batch_reduction='mean'):
        super(FastGradLoss, self).__init__()
        self.ky = torch.tensor([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=torch.float32).reshape(1, 1, 3, 3)
        self.kx = torch.tensor([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=torch.float32).reshape(1, 1, 3, 3)
        self.batch_reduction = batch_reduction
        self.name = 'GradLoss'

    def forward(self, im1, im2):
        from torch.nn.functional import conv2d

        self.kx = self.kx.to(im1.device)
        self.ky = self.ky.to(im1.device)
        im1_gray = torch.mean(im1, 1, keepdim=True)
        im2_gray = torch.mean(im2, 1, keepdim=True)
        diff_x = F.l1_loss(conv2d(im1_gray, self.kx), conv2d(im2_gray, self.kx), reduction=self.batch_reduction)
        diff_y = F.l1_loss(conv2d(im1_gray, self.ky), conv2d(im2_gray, self.ky), reduction=self.batch_reduction)

        if self.batch_reduction == 'none':
            diff_x = torch.mean(diff_x, (1, 2, 3))
            diff_y = torch.mean(diff_y, (1, 2, 3))

        return (diff_x + diff_y) / 2
