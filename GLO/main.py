from GMMN import GMMN
from utils import get_dataloaders
from IMLE import IMLE
from GLO import GLO
from test import run_FID_tests
import torch
import torch.utils.data
import torchvision.utils as vutils

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")


def main():
    train_tag = "-No-z-normalization"
    dataset_name = 'ffhq'
    train_dataloader, test_dataloader, conf = get_dataloaders(dataset_name, device)

    glo = GLO(conf, dataset_size=len(train_dataloader.dataset), device=device)

    train_dir = f"outputs/global-optimizers/{dataset_name}_{glo.loss.name}_{train_tag}"

    # glo.train(train_dataloader, conf, outptus_dir=train_dir, start_epoch=0)
    glo.load_weights(train_dir, device)

    imle = IMLE(conf.e_dim, conf.z_dim, device)
    gmmn = GMMN(conf.e_dim, conf.z_dim, device)
    imle.load_weights(train_dir, device)
    gmmn.load_weights(train_dir, device)
    # Zs = glo.netZ.emb.weight.data
    # gmmn.train(Zs, train_dir=train_dir, epochs=50)
    # imle.train(Zs.cpu().numpy(), train_dir=train_dir, epochs=50)

    z = imle.netT(torch.randn(64, imle.e_dim).to(device))
    z_mmd = imle.netT(torch.randn(64, imle.e_dim).to(device))
    vutils.save_image(glo.netG(z), f"{train_dir}/imgs/IMLE-sampled.png", normalize=True)
    vutils.save_image(glo.netG(z_mmd), f"{train_dir}/imgs/GMMN-sampled.png", normalize=True)


    run_FID_tests(train_dir, glo, imle, gmmn, test_dataloader, train_dataloader, device)


if __name__ == '__main__':
    main()