import os

import torch
import torch.nn as nn

import torchvision.models as models
import torchvision.transforms as transforms
from tqdm import tqdm

from losses.vgg_loss.vgg_loss import get_features_metric, VGGFeatures
from style_transfer.utils import ImageFolder, get_transformer, imload, imsave, calc_loss, calc_TV_Loss
from style_transfer.optimize_style import img_size, crop_size, style_layers, content_layers


class TransformNetwork(nn.Module):
    def __init__(self):        
        super(TransformNetwork, self).__init__()        
        
        self.layers = nn.Sequential(            
            ConvLayer(3, 32, 9, 1),
            ConvLayer(32, 64, 3, 2),
            ConvLayer(64, 128, 3, 2),
            
            ResidualLayer(128, 128, 3, 1),
            ResidualLayer(128, 128, 3, 1),
            ResidualLayer(128, 128, 3, 1),
            ResidualLayer(128, 128, 3, 1),
            ResidualLayer(128, 128, 3, 1),
            
            DeconvLayer(128, 64, 3, 1),
            DeconvLayer(64, 32, 3, 1),
            ConvLayer(32, 3, 9, 1, activation='linear'))
        
    def forward(self, x):
        return self.layers(x)

class ConvLayer(nn.Module):    
    def __init__(self, in_ch, out_ch, kernel_size, stride, pad='reflect', activation='relu', normalization='instance'):        
        super(ConvLayer, self).__init__()
        
        # padding
        if pad == 'reflect':            
            self.pad = nn.ReflectionPad2d(kernel_size//2)
        elif pad == 'zero':
            self.pad = nn.ZeroPad2d(kernel_size//2)
        else:
            raise NotImplementedError("Not expected pad flag !!!")
    
            
        # convolution
        self.conv_layer = nn.Conv2d(in_ch, out_ch, 
                                    kernel_size=kernel_size,
                                    stride=stride)
        
        # activation
        if activation == 'relu':
            self.activation = nn.ReLU()        
        elif activation == 'linear':
            self.activation = lambda x : x
        else:
            raise NotImplementedError("Not expected activation flag !!!")

        # normalization 
        if normalization == 'instance':            
            self.normalization = nn.InstanceNorm2d(out_ch, affine=True)
        else:
            raise NotImplementedError("Not expected normalization flag !!!")

    def forward(self, x):
        x = self.pad(x)
        x = self.conv_layer(x)
        x = self.normalization(x)
        x = self.activation(x)        
        return x
    
class ResidualLayer(nn.Module):    
    def __init__(self, in_ch, out_ch, kernel_size, stride, pad='reflect', normalization='instance'):        
        super(ResidualLayer, self).__init__()
        
        self.conv1 = ConvLayer(in_ch, out_ch, kernel_size, stride, pad, 
                               activation='relu', 
                               normalization=normalization)
        
        self.conv2 = ConvLayer(out_ch, out_ch, kernel_size, stride, pad, 
                               activation='linear', 
                               normalization=normalization)
        
    def forward(self, x):
        y = self.conv1(x)
        return self.conv2(y) + x
        
class DeconvLayer(nn.Module):    
    def __init__(self, in_ch, out_ch, kernel_size, stride, pad='reflect', activation='relu', normalization='instance', upsample='nearest'):        
        super(DeconvLayer, self).__init__()
        
        # upsample
        self.upsample = upsample
        
        # pad
        if pad == 'reflect':            
            self.pad = nn.ReflectionPad2d(kernel_size//2)
        elif pad == 'zero':
            self.pad = nn.ZeroPad2d(kernel_size//2)
        else:
            raise NotImplementedError("Not expected pad flag !!!")        
        
        # conv
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride)
        
        # activation
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'linear':
            self.activation = lambda x : x
        else:
            raise NotImplementedError("Not expected activation flag !!!")
        
        # normalization
        if normalization == 'instance':
            self.normalization = nn.InstanceNorm2d(out_ch, affine=True)
        else:
            raise NotImplementedError("Not expected normalization flag !!!")
        
    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=2, mode=self.upsample)        
        x = self.pad(x)
        x = self.conv(x)
        x = self.normalization(x)        
        x = self.activation(x)        
        return x


def network_train(images_dir, style_img_path, loss_network, lr, max_iter, batch_size, save_path, device):
    """
    Train a translation model that changes the style of its input to the style of the reference style image
    """
    os.makedirs(save_path, exist_ok=True)

    train_dataset = ImageFolder(images_dir, get_transformer(img_size, crop_size))
    target_style_image = imload(style_img_path, imsize=img_size).to(device)
    # target_style_image = cv2pt(cv2.imread(style_img_path), imsize=img_size).to(device)

    # Transform Network
    transform_network = TransformNetwork()
    transform_network = transform_network.to(device)

    # Optimizer
    optimizer = torch.optim.Adam(params=transform_network.parameters(), lr=lr)

    # Target style image
    b, c, h, w = target_style_image.size()
    target_style_image = target_style_image.expand(batch_size, c, h, w)

    # Train
    loss_logs = {'content_loss': [], 'style_loss': [], 'tv_loss': [], 'total_loss': []}

    style_activations = loss_network.get_activations(target_style_image)
    target_style_features = [style_activations[i] for i in style_layers]

    pbar = tqdm(range(max_iter))
    for iteration in pbar:
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        image = next(iter(train_dataloader))
        image = image.to(device)

        output_image = transform_network(image)

        input_activations = loss_network.get_activations(image)
        target_content_features = [input_activations[i] for i in content_layers]

        output_activations = loss_network.get_activations(output_image)
        output_style_features = [output_activations[i] for i in style_layers]
        output_content_features = [output_activations[i] for i in content_layers]

        content_loss = calc_loss(output_content_features, target_content_features, get_features_metric('l2'))
        style_loss = calc_loss(output_style_features, target_style_features, get_features_metric('gram'))
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

            pbar.set_description(
                f"total_loss: {loss_logs['total_loss'][-1]:.1f}, content_loss: {loss_logs['content_loss'][-1]:.1f},"
                f" style_loss: {loss_logs['style_loss'][-1]:.1f}, tv_loss: {loss_logs['tv_loss'][-1]:.1f}")

            imsave(output_image.cpu(), os.path.join(save_path, f"iter-{iteration}.png"))

            torch.save(transform_network.state_dict(), os.path.join(save_path, "transform_network.pth"))

    # save train results
    torch.save(loss_logs, os.path.join(save_path, "loss_logs.pth"))
    torch.save(transform_network.state_dict(), os.path.join(save_path, "transform_network.pth"))

    return transform_network


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    max_iter = 3000
    lr = 0.5
    batch_size = 1
    img_size = 256
    crop_size = 240
    content_layers = ['conv3_3']
    style_layers = ['conv1_2', 'conv2_2', 'conv3_3', 'conv4_3']

    loss_network = VGGFeatures(pretrained=True).to(device)

    train_dir = 'network_outputs/starry_night-pretrained'
    style_img_path = 'imgs/style/starry_night.jpg'
    images_dir = 'dataset'
    network_train(images_dir, style_img_path, loss_network, lr, max_iter, batch_size, train_dir, device)


