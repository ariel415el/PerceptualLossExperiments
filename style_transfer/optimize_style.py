import torch
import os

from matplotlib import pyplot as plt

from losses.vgg_loss.vgg_loss import VGGFeatures, get_features_metric, layer_names_to_indices
from style_transfer.utils import imload, imsave, calc_loss, calc_TV_Loss
from tqdm import tqdm


def style_mix_optimization(content_img_path, style_img_path, loss_network, style_weight, lr, max_iter, save_path, device):
    """
    Optimize an input image that mix style and content of specific two other images
    """
    losses = []
    os.makedirs(save_path, exist_ok=True)
    content_image = imload(content_img_path, imsize=(img_size, img_size)).to(device)
    style_image = imload(style_img_path, imsize=(img_size, img_size)).to(device)

    # target_img = torch.randn(content_image.shape).to(device).float() * 0.5
    # target_img = content_image.clone()
    target_img = torch.ones(style_image.shape).to(device) * torch.mean(style_image.clone(), dim=(2,3), keepdim=True) + torch.randn(content_image.shape).to(device).float() * 0.01
    # target_img = torch.mean(content_image.clone(), dim=(2,3), keepdim=True)
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
        # style_loss = calc_loss(style_activations, target_style_activations, get_features_metric('swd'))
        style_loss = calc_loss(style_activations, target_style_activations, get_features_metric('gram'))

        total_loss = content_loss + style_loss# * style_weight # + calc_TV_Loss(target_img)

        # target_img.data.clamp_(-1, 1)
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
    lr = 0.01
    batch_size = 1
    img_size = 256
    crop_size = 240
    style_weight = 1
    content_layers = []
    # style_layers = ['conv1_2', 'conv2_2', 'conv3_3', 'conv4_3', 'conv5_3']
    style_layers = ['conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', 'conv3_2', 'conv3_3', 'conv4_1', 'conv4_2', 'conv4_3','conv5_1', 'conv5_2', 'conv5_3']
    # style_weight = 1000000
    # content_layers = ['conv2_2']
    # style_layers = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1']


    # loss_network = VGGFeatures(pretrained=False, reinit=True, norm_first_conv=True).to(device)
    loss_network = VGGFeatures(pretrained=True).to(device).eval()
    # loss_network = VGGFeatures(pretrained=False).to(device)
    # loss_network = VGGFeatures(pretrained=True).to(device)

    tag = 'swd'
    style_img_path = 'imgs/style/green_waves.jpg'
    style_img_name = os.path.splitext(os.path.basename(style_img_path))[0]
    content_img_path = 'imgs/content/chicago.jpg'
    content_img_name = os.path.splitext(os.path.basename(content_img_path))[0]
    train_dir = f"outputs/optimize_output_new/{content_img_name}-{style_img_name}-{loss_network.name}-{tag}"
    style_mix_optimization(content_img_path, style_img_path, loss_network, style_weight, lr, max_iter, train_dir, device)
