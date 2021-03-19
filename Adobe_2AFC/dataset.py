import os.path
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import torch
import numpy as np

class TwoAFCDataset(Dataset):
    def __init__(self, dataroots, load_size=64):
        super(TwoAFCDataset, self).__init__()
        self.paths = {'ref':[], 'p0':[], 'p1':[], 'judge':[]}

        # image directory
        for root in dataroots:
            for dirname in self.paths:
                for fn in os.listdir(os.path.join(root, dirname)):
                    self.paths[dirname].append(os.path.join(root, dirname, fn))
                self.paths[dirname] = sorted(self.paths[dirname])

        self.transform = transforms.Compose([transforms.Scale(load_size),
                          transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                          ])


    def __getitem__(self, index):
        p0_img = Image.open(self.paths['p0'][index]).convert('RGB')
        p0_img = self.transform(p0_img)

        p1_img = Image.open(self.paths['p1'][index]).convert('RGB')
        p1_img = self.transform(p1_img)

        ref_img = Image.open(self.paths['ref'][index]).convert('RGB')
        ref_img = self.transform(ref_img)

        label = np.load(self.paths['judge'][index])[0]

        return p0_img, p1_img, ref_img, label

    def __len__(self):
        return len(self.paths['ref'])


def get_dataloader(data_roots, batch_size, num_workers):
    dataset = TwoAFCDataset(data_roots)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers)

    return dataloader