from torchvision import models
import torch
from PIL import Image
import torch.nn.functional
from torchvision import transforms
import numpy as np
import cv2

vgg = models.vgg16()
vgg.load_state_dict(torch.load("vgg16-397923af.pth"))

transform = transforms.Compose([            #[1]
 transforms.Resize(256),                    #[2]
 transforms.CenterCrop(224),                #[3]
 transforms.ToTensor(),                     #[4]
 transforms.Normalize(                      #[5]
 mean=[0.485, 0.456, 0.406],                #[6]
 std=[0.229, 0.224, 0.225]                  #[7]
 )])

# img = Image.open("dog.jpg")
# img = transform(img)
img = cv2.cvtColor(cv2.imread("dog.jpg"), cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (224,224))
img = img.transpose(2,0,1).astype(np.float32) / 255
img = torch.from_numpy(img)
img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
# img = 2 * img - 1
batch_t = torch.unsqueeze(img, 0)
vgg.eval()
out = vgg(batch_t)
_, index = torch.max(out, 1)
percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
print(index[0], percentage[index[0]].item())
