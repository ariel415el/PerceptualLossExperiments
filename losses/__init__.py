
from losses.ssim.SSIM import SSIM

from losses.empty_loss import *
from losses.experimental_patch_losses import *

from losses.classic_losses.l2 import L2
from losses.classic_losses.grad_loss import GradLoss, GradLoss3Channels
from losses.classic_losses.lap1_loss import LapLoss
from losses.composite_losses.list_loss import LossesList
from losses.composite_losses.pyramid_loss import PyramidLoss
from losses.experimental_patch_losses import MMD_PP
from losses.mmd.patch_mmd import *
from losses.mmd.approximate_patch_mmd import MMDApproximate
from losses.local_term_loss import PatchRBFLoss
from losses.vgg_loss.vgg_loss import VGGPerceptualLoss
from losses.alexnet_loss.alexnet_loss import AlexNetLoss
from losses.patch_coherence_loss import PatchCoherentLoss
