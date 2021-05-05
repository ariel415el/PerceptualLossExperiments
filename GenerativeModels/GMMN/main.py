import os
import torch.utils.data


import sys
sys.path.append(os.path.realpath("../.."))
from GenerativeModels.GMMN.GMMN import GMMN
from GenerativeModels.utils.data_utils import get_dataset
from GenerativeModels.models import DCGANGenerator

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")


def main():
    latent_dim = 64
    # dataset_name, img_dim, channels = 'mnist', 28, 1
    dataset_name, img_dim, channels = 'ffhq', 64, 3
    train_dir = f"outputs/{dataset_name}-64-latent_dim-vgg-5-PreRelu-all_data"

    train_dataset = get_dataset(dataset_name, split='train')

    print(f"Dataset size: {len(train_dataset)}")

    # define the generator
    generator = DCGANGenerator(latent_dim, channels, img_dim)

    gmmn = GMMN(generator, lr=0.0001, batch_size=128, device=device)

    gmmn.train(train_dataset, train_dir, epochs=200)


if __name__ == '__main__':
    main()