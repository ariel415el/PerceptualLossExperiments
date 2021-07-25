from tqdm import tqdm
import numpy as np
import torch

from dataset import get_dataloader

# from losses.mmd_exact_loss import MMDExact
from losses.mmd.windowed_patch_mmd import MMDApproximate
from losses.vgg_loss.vgg_loss import VGGPerceptualLoss

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

def mse_looss(x,y):
    return torch.mean((x.view(x.size(0),-1)-y.view(x.size(0),-1))**2, dim=1)


def score_2afc_dataset(data_loader, func, name=''):
    ''' Function computes Two Alternative Forced Choice (2AFC) score using
        distance function 'func' in dataset 'data_loader'
    INPUTS
        data_loader - CustomDatasetDataLoader object - contains a TwoAFCDataset inside
        func - callable distance function - calling d=func(in0,in1) should take 2
            pytorch tensors with shape Nx3xXxY, and return numpy array of length N
    OUTPUTS
        [0] - 2AFC score in [0,1], fraction of time func agrees with human evaluators
        [1] - dictionary with following elements
            d0s,d1s - N arrays containing distances between reference patch to perturbed patches
            gts - N array in [0,1], preferred patch selected by human evaluators
                (closer to "0" for left patch p0, "1" for right patch p1,
                "0.6" means 60pct people preferred right patch, 40pct preferred left)
            scores - N array in [0,1], corresponding to what percentage function agreed with humans
    CONSTS
        N - number of test triplets in data_loader
    '''

    d0s = []
    d1s = []
    gts = []

    for d0_batchs, d1_batchs, refs_batch, label_batch in tqdm(data_loader, desc=name):
        d0_batchs, d1_batchs, refs_batch = d0_batchs.to(device), d1_batchs.to(device), refs_batch.to(device)
        d0s += func(refs_batch, d0_batchs).detach().cpu().numpy().tolist()
        d1s += func(refs_batch, d1_batchs).detach().cpu().numpy().tolist()
        gts += label_batch.tolist()

    d0s = np.array(d0s)
    d1s = np.array(d1s)
    gts = np.array(gts)

    scores = (d0s < d1s) * (1. - gts) + (d1s < d0s) * gts + (d1s == d0s) * .5

    return np.mean(scores)


def main():
    criterions = [
        # MMD_PP(batch_reduction='none'),
        # VGGPerceptualLoss(pretrained=False, batch_reduction='none'),
        # VGGPerceptualLoss(pretrained=False, reinit=True, batch_reduction='none'),
        VGGPerceptualLoss(pretrained=False, reinit=False, norm_first_conv=True, batch_reduction='none')
        # MMDApproximate(patch_size=3, sigma=0.06, pool_size=32, pool_strides=16, r=64, batch_reduction='none', normalize_patch='channel_mean'),
        # MMDApproximate(patch_size=3, sigma=0.1, pool_size=32, pool_strides=16, r=64, batch_reduction='none', normalize_patch='channel_mean'),
        # MMDApproximate(patch_size=3, sigma=0.06, pool_size=16, pool_strides=8, r=64, batch_reduction='none', normalize_patch='channel_mean'),
        # MMDApproximate(patch_size=3, sigma=0.06, pool_size=8, pool_strides=4, r=64, batch_reduction='none', normalize_patch='channel_mean'),
    ]

    dataloader = get_dataloader([
                                "../../../data/Perceptual2AFC/2afc/val/cnn",
                                 "../../../data/Perceptual2AFC/2afc/val/traditional"
                                ],
                                batch_size=8,
                                num_workers=4)

    f = open("results.txt", 'a')
    f.write(f"criterion: mean/std\n")
    for criterion in criterions:
        scores = []
        for i in range(1):
            scores.append(score_2afc_dataset(dataloader, criterion.to(device)))
        f.write(f"{criterion.name}: {np.mean(scores):.3f}/{np.var(scores):.2f}\n")
        print(f"{criterion.name}: {np.mean(scores):.3f}/{np.var(scores):.2f}")
    f.close()

if __name__ == '__main__':
    main()
