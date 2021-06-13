import torch
import os

from matplotlib import pyplot as plt

from losses.vgg_loss.vgg_loss import VGGFeatures, get_features_metric, layer_names_to_indices
from style_transfer.utils import imload, imsave, calc_loss, calc_TV_Loss
from tqdm import tqdm


def style_mix_optimization(content_img_path, style_img_path, loss_network, lr, max_iter, save_path, device):
    """
    Optimize an input image that mix style and content of specific two other images
    """
    losses = []
    os.makedirs(save_path, exist_ok=True)
    content_image = imload(content_img_path, imsize=(img_size, img_size)).to(device)
    style_image = imload(style_img_path, imsize=(img_size, img_size)).to(device)

    # target_img = torch.randn(content_image.shape).to(device).float() * 0.5
    target_img = content_image.clone()
    target_img.requires_grad_(True)

    optimizer = torch.optim.Adam([target_img], lr=lr)

    style_activations = loss_network.get_activations(style_image)
    style_activations = [style_activations[x] for x in style_layers]

    content_activations = loss_network.get_activations(content_image)
    content_activations = [content_activations[x] for x in content_layers]

    pbar = tqdm(range(max_iter + 1))
    for iteration in pbar:
        if iteration % 500 == 0:
            imsave(target_img.cpu(), os.path.join(save_path, f"iter-{iteration}.png"))
            plt.plot(range(len(losses)), losses)
            plt.savefig(os.path.join(save_path+".png"))
            plt.clf()
        target_activations = loss_network.get_activations(target_img)
        target_style_activations = [target_activations[x] for x in style_layers]
        target_content_activations = [target_activations[x] for x in content_layers]

        # content_loss = calc_loss(content_activations, target_content_activations, get_features_metric('cx', h=0.2))
        # style_loss = calc_loss(style_activations, target_style_activations, get_features_metric('cx', h=0.1))
        content_loss = calc_loss(content_activations, target_content_activations, get_features_metric('l2'))
        style_loss = calc_loss(style_activations, target_style_activations, get_features_metric('gram'))

        total_loss = content_loss + style_loss * 30 + calc_TV_Loss(target_img)

        optimizer.zero_grad()
        total_loss.backward(retain_graph=True)
        optimizer.step()
        # print loss logs
        losses.append(total_loss.item())
        if iteration % 250 == 0:
            for g in optimizer.param_groups:
                g['lr'] *= 0.75
            pbar.set_description(f"total_loss: {total_loss}")


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    max_iter = 1000
    lr = 0.5
    batch_size = 1
    img_size = 256
    crop_size = 240
    content_layers = ['conv3_3']
    style_layers = ['conv1_2', 'conv2_2', 'conv3_3', 'conv4_3']
    # content_layers = ['conv4_2']
    # style_layers = ['conv3_2', 'conv4_2']

    # import torchvision
    # loss_network = torchvision.models.__dict__['vgg16'](pretrained=True).features.to(device)

    # loss_network = VGGFeatures(pretrained=False, reinit=True, norm_first_conv=True).to(device)
    loss_network = VGGFeatures(pretrained=False, norm_first_conv=True).to(device)
    # loss_network = VGGFeatures(pretrained=False).to(device)
    # loss_network = VGGFeatures(pretrained=True).to(device)

    train_dir = f"outputs/optimize_output_new/abstract-cornel-{loss_network.name}"
    style_img_path = 'imgs/style/abstraction.jpg'
    # style_img_path = 'imgs/faces/00001.png'
    # content_img_path = 'imgs/faces/00006.png'
    content_img_path = 'imgs/content/cornell.jpg'
    style_mix_optimization(content_img_path, style_img_path, loss_network, lr, max_iter, train_dir, device)
