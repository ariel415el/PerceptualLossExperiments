from losses.SSIM_1 import SSIM
from losses.SSIM_2 import SSIM_2
from losses.empty_loss import *
from losses.experimental_patch_losses import *

from losses.classic_losses.l2 import L2
from losses.classic_losses.grad_loss import GradLoss, GradLoss3Channels
from losses.classic_losses.lap1_loss import LapLoss
from losses.composite_losses.list_loss import LossesList
from losses.composite_losses.pyramid_loss import PyramidLoss
from losses.experimental_patch_losses import MMD_PP
from losses.mmd.patch_mmd import PatchMMDLoss
from losses.mmd.windowed_patch_mmd import MMDApproximate
from losses.patch_loss import PatchRBFLoss
from losses.vgg_loss.vgg_loss import VGGPerceptualLoss
from losses.alexnet_loss.alexnet_loss import AlexNetLoss
