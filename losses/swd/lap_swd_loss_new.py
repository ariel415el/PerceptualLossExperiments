import torch
from losses.swd.swd import PatchSWDLoss
from losses.composite_losses.laplacian_losses import LaplacyanLoss

class LapSWDLoss_new(torch.nn.Module):
    def __init__(self, batch_reduction='mean'):
        super(LapSWDLoss_new, self).__init__()
        self.batch_reduction = batch_reduction

        self.loss = LaplacyanLoss(PatchSWDLoss(patch_size=7, num_proj=512, n_samples=512, batch_reduction=batch_reduction), weightening_mode=3, max_levels=2)
        self.name = 'LapSWD_new'

    def forward(self, x, y):
        return self.loss(x,y)
