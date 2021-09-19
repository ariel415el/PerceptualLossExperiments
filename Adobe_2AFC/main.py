from tqdm import tqdm
import numpy as np
import torch

from dataset import get_dataloader

import losses

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
        # losses.VGGPerceptualLoss(pretrained=False, reinit=True, norm_first_conv=True, layers_and_weights=[('conv1_2', 1)], batch_reduction='none', name='vgg-rand_conv1_2'),
        # losses.VGGPerceptualLoss(pretrained=False, reinit=True, norm_first_conv=True, layers_and_weights=[('conv2_2', 1)], batch_reduction='none', name='vgg-rand_conv2_2'),
        # losses.VGGPerceptualLoss(pretrained=False, reinit=True, norm_first_conv=True, layers_and_weights=[('conv3_3', 1)], batch_reduction='none', name='vgg-rand_conv3_3'),
        # losses.VGGPerceptualLoss(pretrained=False, reinit=True, norm_first_conv=True, layers_and_weights=[('conv4_3', 1)], batch_reduction='none', name='vgg-rand_conv4_3'),
        # losses.VGGPerceptualLoss(pretrained=False, reinit=True, norm_first_conv=True, layers_and_weights=[('conv5_3', 1)], batch_reduction='none', name='vgg-rand_conv5_3'),
        #
        # losses.VGGPerceptualLoss(pretrained=True, layers_and_weights=[('conv1_2', 1)], batch_reduction='none', name='vgg-conv1_2'),
        # losses.VGGPerceptualLoss(pretrained=True, layers_and_weights=[('conv2_2', 1)], batch_reduction='none', name='vgg-conv2_2'),
        # losses.VGGPerceptualLoss(pretrained=True, layers_and_weights=[('conv3_3', 1)], batch_reduction='none', name='vgg-conv3_3'),
        # losses.VGGPerceptualLoss(pretrained=True, layers_and_weights=[('conv4_3', 1)], batch_reduction='none', name='vgg-conv4_3'),
        # losses.VGGPerceptualLoss(pretrained=True, layers_and_weights=[('conv5_3', 1)], batch_reduction='none', name='vgg-onv5_3'),
        losses.SSIM(batch_reduction='none')
    ]

    dataloader = get_dataloader([
                                "../../../data/Perceptual2AFC/2afc/val/cnn",
                                 "../../../data/Perceptual2AFC/2afc/val/traditional"
                                ],
                                batch_size=8,
                                num_workers=0)

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
