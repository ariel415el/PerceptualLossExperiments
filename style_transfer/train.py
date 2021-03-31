import time

import torch
import os
from losses.vgg_loss.vgg_loss import VGGFeatures
from network import TransformNetwork
from image_utils import ImageFolder, get_transformer, imload, imsave
from tqdm import tqdm

mse_criterion = torch.nn.MSELoss(reduction='mean')

def calc_Content_Loss(features, targets, weights=None):
    if weights is None:
        weights = [1/len(features)] * len(features)
    
    content_loss = 0
    for f, t, w in zip(features, targets, weights):
        content_loss += mse_criterion(f, t) * w
        
    return content_loss


def gram(x):
    b ,c, h, w = x.size()
    g = torch.bmm(x.view(b, c, h*w), x.view(b, c, h*w).transpose(1,2))
    return g.div(h*w)


def calc_Gram_Loss(features, targets, weights=None):
    if weights is None:
        weights = [1/len(features)] * len(features)
        
    gram_loss = 0
    for f, t, w in zip(features, targets, weights):
        gram_loss += mse_criterion(gram(f), gram(t)) * w
    return gram_loss


def calc_TV_Loss(x):
    tv_loss = torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]))
    tv_loss += torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
    return tv_loss


def extract_features(model, x, layers):
    features = list()
    for index, layer in enumerate(model):
        x = layer(x)
        if index in layers:
            features.append(x)
    return features


def network_train(images_dir, style_img_path, loss_network, lr, max_iter, batch_size, save_path, device):
    os.makedirs(save_path, exist_ok=True)

    train_dataset = ImageFolder(images_dir, get_transformer(img_size, crop_size))
    target_style_image = imload(style_img_path, imsize=img_size).to(device)

    # Transform Network
    transform_network = TransformNetwork()
    transform_network = transform_network.to(device)

    # Optimizer
    optimizer = torch.optim.Adam(params=transform_network.parameters(), lr=lr)

    # Target style image
    b, c, h, w = target_style_image.size()
    target_style_image = target_style_image.expand(batch_size, c, h, w)

    # Train
    loss_logs = {'content_loss':[], 'style_loss':[], 'tv_loss':[], 'total_loss':[]}

    style_activations = loss_network.get_activations(target_style_image)
    target_style_features = [style_activations[i] for i in style_layers]
    # target_style_features = extract_features(loss_network, target_style_image, style_layers)

    pbar = tqdm(range(max_iter))
    for iteration in pbar:
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        image = next(iter(train_dataloader))
        image = image.to(device)

        output_image = transform_network(image)

        input_activations = loss_network.get_activations(image)
        target_content_features = [input_activations[i] for i in content_layers]
        # target_content_features = extract_features(loss_network, image, content_layers)

        output_activations = loss_network.get_activations(output_image)
        output_style_features = [output_activations[i] for i in style_layers]
        output_content_features = [output_activations[i] for i in content_layers]
        # output_style_features = extract_features(loss_network, output_image, style_layers)
        # output_content_features = extract_features(loss_network, output_image, content_layers)

        content_loss = calc_Content_Loss(output_content_features, target_content_features)
        style_loss = calc_Gram_Loss(output_style_features, target_style_features)
        tv_loss = calc_TV_Loss(output_image)
        
        total_loss = content_loss + style_loss * 30 + tv_loss

        loss_logs['content_loss'].append(content_loss.item())
        loss_logs['style_loss'].append(style_loss.item())
        loss_logs['tv_loss'].append(tv_loss.item())
        loss_logs['total_loss'].append(total_loss.item())

        optimizer.zero_grad()
        total_loss.backward(retain_graph=True)
        optimizer.step()

        # print loss logs
        if iteration % 250 == 0:
            for g in optimizer.param_groups:
                g['lr'] *= 0.99

            pbar.set_description(f"total_loss: {loss_logs['total_loss'][-1]:.1f}, content_loss: {loss_logs['content_loss'][-1]:.1f},"
                                 f" style_loss: {loss_logs['style_loss'][-1]:.1f}, tv_loss: {loss_logs['tv_loss'][-1]:.1f}")

            imsave(output_image.cpu(), os.path.join(save_path, f"iter-{iteration}.png"))

            torch.save(transform_network.state_dict(), os.path.join(save_path, "transform_network.pth"))

    # save train results
    torch.save(loss_logs, os.path.join(save_path, "loss_logs.pth"))
    torch.save(transform_network.state_dict(), os.path.join(save_path, "transform_network.pth"))

    return transform_network

def style_mix_optimization(content_img_path, style_img_path, loss_network, lr, max_iter, save_path, device):
    os.makedirs(save_path, exist_ok=True)
    content_image = imload(content_img_path, imsize=img_size).to(device)
    style_image = imload(style_img_path, imsize=img_size).to(device)

    # target_img = torch.randn(content_image.shape)
    target_img = content_image.clone()

    target_img.requires_grad_(True)

    optimizer = torch.optim.Adam([target_img], lr=lr)

    content_activations = loss_network.get_activations(content_image)
    content_activations = [content_activations[i] for i in content_layers]

    style_activations = loss_network.get_activations(style_image)
    style_activations = [style_activations[i] for i in style_layers]

    pbar = tqdm(range(max_iter))
    for iteration in pbar:
        target_activations = loss_network.get_activations(target_img)
        target_style_activations = [target_activations[i] for i in style_layers]
        target_content_activations = [target_activations[i] for i in content_layers]

        content_loss = calc_Content_Loss(content_activations, target_content_activations)
        style_loss = calc_Gram_Loss(style_activations, target_style_activations)
        tv_loss = calc_TV_Loss(target_img)

        total_loss = content_loss + style_loss * 30 + tv_loss

        optimizer.zero_grad()
        total_loss.backward(retain_graph=True)
        optimizer.step()
        # print loss logs
        if iteration % 250 == 0:
            for g in optimizer.param_groups:
                g['lr'] *= 0.95

            pbar.set_description(f"total_loss: {total_loss}")

            imsave(target_img.cpu(), os.path.join(save_path, f"iter-{iteration}.png"))


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    max_iter = 15000
    lr = 0.1
    batch_size = 6
    img_size = 256
    crop_size = 240
    # content_layers = [15]
    content_layers = [2]
    # style_layers = [3, 8, 15, 22]
    style_layers = [0, 1, 2, 3]

    # import torchvision
    # loss_network = torchvision.models.__dict__['vgg16'](pretrained=True).features.to(device)

    loss_network = VGGFeatures(4, pretrained=True).to(device)

    train_dir = 'network_outputs/starry_night-pretrained'
    style_img_path = 'imgs/style/starry_night.jpg'
    images_dir = 'dataset'
    network_train(images_dir, style_img_path, loss_network, lr, max_iter, batch_size, train_dir, device)

    # train_dir = 'optimize_output/home_alone-bradd_pitt_pretrained'
    # style_img_path = 'imgs/style/yellow_sunset.jpg'
    # content_img_path = 'imgs/content/home_alone.jpg'
    # style_mix_optimization(content_img_path, style_img_path, loss_network, lr, max_iter, train_dir, device)