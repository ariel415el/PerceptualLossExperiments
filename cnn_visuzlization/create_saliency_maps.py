import os

import numpy as np
import torch
import torch.nn.functional as F

from cnn_visuzlization.common import save_scaled_images


class VanillaBackprop:
    """
        Produces gradients generated with vanilla back propagation from the image
    """
    def __init__(self, model):
        self.model = model
        self.gradients = None
        # Put model in evaluation mode
        self.model.eval()
        # Hook the first layer to get the gradient
        self.hook_layers()

    def hook_layers(self):
        def hook_function(module, grad_in, grad_out):
            self.gradients = grad_in[0]

        # Register hook to the first layer
        first_layer = list(self.model.features._modules.items())[0][1]
        first_layer.register_backward_hook(hook_function)

    def generate_gradients(self, input_image, target_class):
        # Forward
        model_output = self.model(input_image)
        # Zero grads
        self.model.zero_grad()
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1
        # Backward pass
        model_output.backward(gradient=one_hot_output)
        # Convert Pytorch variable to numpy array
        # [0] to get rid of the first channel (1,3,224,224)
        gradients_as_arr = self.gradients.data.numpy()[0]
        return gradients_as_arr


def show_saliency_maps(net, dataloader, resize_patch, output_dir, device):
    os.makedirs(output_dir, exist_ok=True)
    for c_idx in range(0, net.m_n_maps, 10):
        image = torch.from_numpy(dataloader.dataset[np.random.randint(0, len(dataloader.dataset))][1]).to(device).float().unsqueeze(0)

        all_patches = F.unfold(image, kernel_size=net.m_receptive_field, padding=0, stride=net.m_stride)
        patch_idx = np.random.choice(all_patches.shape[-1])
        patch = all_patches[0, :, patch_idx].reshape(3, net.m_receptive_field, net.m_receptive_field)

        VBP = VanillaBackprop(net)
        vanilla_grads = VBP.generate_gradients(patch, c_idx)

        save_scaled_images(vanilla_grads, resize_patch, f"{output_dir}/c-{c_idx}_grads.png")
        save_scaled_images(image, 1, f"{output_dir}/c-{c_idx}_img.png")
        save_scaled_images(patch, resize_patch, f"{output_dir}/c-{c_idx}_patch.png")