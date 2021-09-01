import torch
import torch.nn as nn
from torchvision.utils import save_image


class myNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, (3,3), stride=(1,1), padding=0, bias=True)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 128, (3,3), stride=(1,1), padding=0, bias=True)
        # self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        return x

gradient_map = None

def hook_fn(m, i, o):
    print(m)
    print([grad.shape for grad in i])
    print([grad.shape for grad in o])
    print(i[0].mean())
    print("----------------------------")
    global gradient_map
    gradient_map = i[0]


# net = myNet()
# net.conv1.register_backward_hook(hook_fn)
# inp = torch.randn(1, 3, 5, 5)

from cnn_visuzlization.common import get_single_conv_vgg
net = get_single_conv_vgg([64, 64, 'M', 128],#, 128],#, 'M', 256],#, 256, 256, 'M', 512, 512, 512],
                          load_weights=True, inplace_relus=False)
list(net._modules.items())[0][1].register_full_backward_hook(hook_fn)


maps = []
images = [torch.randn(1, 3, 24, 24),
            torch.load('images/hair.pt').cpu().unsqueeze(0),
            torch.load('images/mouth.pt').cpu().unsqueeze(0),
            torch.load('images/eye.pt').cpu().unsqueeze(0)]

for inp in [torch.randn(1, 3, 24, 24),
            # torch.load('images/image.pt').cpu(),
            torch.load('images/hair.pt').cpu().unsqueeze(0),
            torch.load('images/mouth.pt').cpu().unsqueeze(0),
            torch.load('images/eye.pt').cpu().unsqueeze(0)]:

    inp.requires_grad_(True)
    out = net(inp)

    one_hot_output = torch.FloatTensor(out.shape).zero_()
    one_hot_output[0][25] = 1

    # Backward pass
    out.backward(gradient=one_hot_output)
    maps.append(gradient_map.clone())

save_image(torch.cat(maps), "maps.png", normalize=True)
save_image(torch.cat(images), "images.png", normalize=True)