import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from cnn_visuzlization.common import save_scaled_images


class VanillaBackprop:
    """
        Produces gradients generated with vanilla back propagation from the image
    """
    def __init__(self, model, guided=False):
        self.model = model
        self.gradients = None
        # Put model in evaluation mode
        self.model.eval()
        # Hook the first layer to get the gradient
        if guided:
            self.forward_relu_outputs = []
            self.update_relus()
        self.hook_layers()

    def hook_layers(self):
        def hook_function(module, grad_in, grad_out):
            # import pydevd
            # pydevd.settrace(suspend=False, trace_only_current_thread=True)
            self.gradients = grad_in[0]

        # Register hook to the first layer
        first_layer = list(self.model._modules.items())[0][1]
        first_layer.register_full_backward_hook(hook_function)

    def update_relus(self):
        """
            Updates relu activation functions so that
                1- stores output in forward pass
                2- imputes zero for gradient values that are less than zero
        """
        def relu_backward_hook_function(module, grad_in, grad_out):
            """
            If there is a negative gradient, change it to zero
            """
            # Get last forward output
            corresponding_forward_output = self.forward_relu_outputs[-1]
            corresponding_forward_output[corresponding_forward_output > 0] = 1
            modified_grad_out = corresponding_forward_output * torch.clamp(grad_in[0], min=0.0)
            del self.forward_relu_outputs[-1]  # Remove last forward output
            return (modified_grad_out,)

        def relu_forward_hook_function(module, ten_in, ten_out):
            """
            Store results of forward pass
            """
            self.forward_relu_outputs.append(ten_out)

        # Loop through layers, hook up ReLUs
        for pos, module in self.model._modules.items():
            if isinstance(module, torch.nn.ReLU):
                module.register_backward_hook(relu_backward_hook_function)
                module.register_forward_hook(relu_forward_hook_function)

    def generate_gradients(self, input_image, c):
        # Inputs need to be the variable w.r.t which we compute gradients
        input_image = Variable(input_image, requires_grad=True)

        model_output = self.model(input_image)

        self.model.zero_grad()

        # Gradients are zeroed in all channels but the desired one
        one_hot_output = torch.FloatTensor(model_output.shape).zero_().to(input_image.device)
        one_hot_output[:,c] = 1
        model_output.backward(gradient=one_hot_output)

        # Return the gradietns saved by the backward hook function
        return self.gradients.data


def show_saliency_maps(net, dataloader, resize_patch, output_dir, device, n_images=10, n_channels=10):
    VBP = VanillaBackprop(net, guided=True)
    image_indices = np.random.choice(range(len(dataloader.dataset)), n_images, replace=False)
    c_indices = np.random.choice(range(net.m_n_maps), n_channels, replace=False)
    # image_indices = [154]
    # c_indices = [37]
    for img_idx in image_indices:
        img_dir = f"{output_dir}/{img_idx}"
        image = torch.from_numpy(dataloader.dataset[img_idx][1]).to(device).float().unsqueeze(0)
        all_patches = F.unfold(image, kernel_size=net.m_receptive_field, padding=0, stride=net.m_stride)
        all_patches = all_patches[0].transpose(1, 0).reshape(-1, 3, net.m_receptive_field, net.m_receptive_field)

        activations = net(image).detach().transpose(1, 0).repeat(1, 3, 1, 1)

        save_scaled_images(activations, resize_patch, f"{img_dir}/all_activations.png")
        save_scaled_images(image, resize_patch, f"{img_dir}/image.png")
        save_scaled_images(all_patches.clone(), resize_patch, f"{img_dir}/all_patches.png")

        for c_idx in c_indices:
            image_grads = VBP.generate_gradients(image, c_idx)
            patch_grads = VBP.generate_gradients(all_patches, c_idx)

            save_scaled_images(image_grads, resize_patch, f"{img_dir}/channels/c-{c_idx}_image_grads.png")
            save_scaled_images(patch_grads, resize_patch, f"{img_dir}/channels/c-{c_idx}_patch_grads.png")
            save_scaled_images(activations[c_idx], resize_patch, f"{img_dir}/channels/c-{c_idx}_activations.png")

            from torchvision.transforms import transforms
            no_padding_image_grads = image_grads[:, :, net.m_receptive_field//2:-net.m_receptive_field//2, net.m_receptive_field//2:-net.m_receptive_field//2]
            weighted_image_grads = transforms.Resize((no_padding_image_grads.shape[2], no_padding_image_grads.shape[3]), antialias=True)(activations[c_idx])
            save_scaled_images(no_padding_image_grads * weighted_image_grads.unsqueeze(0), resize_patch, f"{img_dir}/channels/c-{c_idx}_scaled_image_grads.png")

if __name__ == '__main__':
    import cv2
    from perceptual_mean_optimization.utils import cv2pt

    from torchvision import transforms
    from torchvision.transforms import InterpolationMode
    from torchvision import utils as vutils

    img = cv2.imread('/home/ariel/university/GPDM/GPDM/images/resampling/balloons.png')

    all_patches = F.unfold(cv2pt(img).unsqueeze(0), kernel_size=32, padding=0, stride=16)
    all_patches = all_patches[0].transpose(1, 0).reshape(-1, 3, 32, 32)
    all_patches = transforms.Resize(all_patches.shape[-1] * 5, interpolation=InterpolationMode.NEAREST)(all_patches)

    indices = np.random.randint(0, all_patches.shape[0], size=10)

    vutils.save_image(all_patches, 'target.png', normalize=True, nrow=img.shape[1] // 16 - 1, pad_value=64, padding=10)

    for i in indices:
        save_scaled_images(all_patches[i], 1, f"target{i}.png")

    img = cv2.resize(
        cv2.imread('/home/ariel/university/GPDM/GPDM/outputs/image_resampling/balloons/PatchSWD(p-11:1)_AR-(1.0, 1.0)_R-256_S-0.75x5_I-noise##/0/output-500.png')
        , (img.shape[1], img.shape[0]))
    # img = cv2.imread('/home/ariel/university/GPDM/GPDM/images/retargeting/fruit.png')
    img = cv2pt(img).unsqueeze(0)
    all_patches = F.unfold(img, kernel_size=32, padding=0, stride=16)
    all_patches = all_patches[0].transpose(1, 0).reshape(-1, 3, 32, 32)
    all_patches = transforms.Resize(all_patches.shape[-1] * 5, interpolation=InterpolationMode.NEAREST)(all_patches)
    vutils.save_image(all_patches, 'img.png', normalize=True, nrow=img.shape[-1] // 16 - 1, pad_value=64, padding=10)

    for i in indices:
        save_scaled_images(all_patches[i], 1, f"img{i}.png")