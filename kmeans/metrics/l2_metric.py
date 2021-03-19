import numpy as np
import torch

class L2Metric:
    def __init__(self):
        self.name = f"L2"
        self.feature_extractor = Flatter()
        self.metric = lambda x,y: np.mean((x-y)**2, axis=-1)


class Flatter(torch.nn.Module):
    def __init__(self):
        super(Flatter, self).__init__()

    def get_fv(self, x):
        return x.reshape(x.shape[0], -1)