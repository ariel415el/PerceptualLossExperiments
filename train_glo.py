from glo import GLO
from config import mnist_configs, faces_config
import torch
from utils import MnistDataset, DiskDataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main():
    # train_samples_path = "../data/fashionMnist/train_fashion.npy"
    # train_samples_path = "../data/Mnist/train_Mnist.npy"
    # dataset = MnistDataset(train_samples_path)
    # conf = mnist_configs
    # train_dir = "training_dir/FMNIST-vgg_loss"
    # train_dir = "training_dir/MNIST-vgg_loss"

    train_samples_path = "../data/img_align_celeba"
    dataset = DiskDataset(train_samples_path)
    conf = faces_config
    train_dir = "training_dir/celebA-vgg_loss_2"

    glo = GLO(conf, dataset_size=len(dataset), device=device)
    glo.train(dataset, conf, outptus_dir=train_dir)


if __name__ == '__main__':
    main()