import os
from dataclasses import dataclass
from typing import Tuple
import losses
from image_retargeting.retarget_image import retarget_image

def get_file_name(path):
    return os.path.splitext(os.path.basename(path))[0]

@dataclass
class SyntesisConfigurations:
    aspect_ratio: Tuple[float, float] = (1.,1.)
    resize: int = 256
    pyr_factor: float = 0.7
    n_scales: int = 5
    lr: float = 0.05
    num_steps: int = 500
    init: str = 'noise'
    blur_loss: float = 0.0
    tv_loss: float = 0.0
    device: str = 'cuda:0'

    def get_conf_tag(self):
        init_name = 'img' if os.path.exists(self.init) else self.init
        if self.blur_loss > 0:
            init_name += f"_BL({self.blur_loss})"
        if self.tv_loss > 0:
            init_name += f"_TV({self.tv_loss})"
        return f'AR-{self.aspect_ratio}_R-{self.resize}_S-{self.pyr_factor}x{self.n_scales}_I-{init_name}'


def retarget_penguins():
    input_image_path = 'images/resampling/pinguins.png'

    # criteria = losses.MMDApproximate(patch_size=11, strides=1, sigma=0.03, pool_size=-1, r=512, normalize_patch='channel_mean')
    criteria = losses.PatchSWDLoss(patch_size=11, stride=1, num_proj=512, normalize_patch='mean')

    conf = SyntesisConfigurations(pyr_factor=0.6, n_scales=5, aspect_ratio=(1.5,0.6), lr=0.05, num_steps=500, init="blur", resize=256)

    outputs_dir = f'test_outputs/image_retargeting/{get_file_name(input_image_path)}_AR-{conf.aspect_ratio}/{criteria.name}_{conf.get_conf_tag()}'

    retarget_image(input_image_path, criteria, conf, outputs_dir)


def retartget():
    for aspect_ratio in [(1,0.5), (1, 1.5)]:
        # input_image_path = 'images/resampling/pinguins.png'
        # input_image_path = 'images/retargeting/SupremeCourt.jpeg'
        # input_image_path = 'images/retargeting/corn.png'
        # input_image_path = 'images/resampling/balloons.png'
        # input_image_path = 'images/resampling/birds.png'
        # input_image_path = 'images/resampling/girafs.png'
        # input_image_path = 'images/retargeting/fruit.png'
        # input_image_path = 'images/retargeting/kanyon.jpg'
        input_image_path = 'images/retargeting/fish.png'

        # criteria = losses.MMDApproximate(patch_size=7, strides=1, sigma=0.03, pool_size=-1, r=1024, normalize_patch='channel_mean')
        criteria = losses.PatchSWDLoss(patch_size=11, stride=1, num_proj=1024, normalize_patch='mean')

        conf = SyntesisConfigurations(pyr_factor=0.6, n_scales=4, aspect_ratio=aspect_ratio, lr=0.05, num_steps=500, init="blur", resize=256)

        outputs_dir = f'test_outputs/image_retargeting/{get_file_name(input_image_path)}/{criteria.name}_{conf.get_conf_tag()}'

        retarget_image(input_image_path, criteria, conf, outputs_dir)


def resample():
    for input_image_path in [
        'images/resampling/cows.png',
        # 'images/resampling/pool1.jpg',
        # 'images/resampling/balloons.png',
        # 'images/resampling/solar_system2.jpg',
        # 'images/resampling/planes1.jpg',
        # 'images/resampling/people_on_the_beach.jpg',
        # 'images/resampling/balls.jpg',
        # 'images/resampling/green_view.jpg',
        # 'images/resampling/soccer1.png',
        # 'images/resampling/birds.png',
        # 'images/resampling/soccer3.jpg',
        # 'images/resampling/jerusalem2.jpg',
        # 'images/resampling/soccer2.jpg'
    ]:
        # criteria = losses.MMDApproximate(patch_size=5, strides=1, sigma=0.03, pool_size=-1, r=512, normalize_patch='channel_mean')
        criteria = losses.PatchSWDLoss(patch_size=5, stride=1, num_proj=1024, normalize_patch='none')

        conf = SyntesisConfigurations(pyr_factor=0.75, n_scales=5, lr=0.05, num_steps=1000, init="noise", resize=256)

        outputs_dir = f'test_outputs/resampling/{get_file_name(input_image_path)}/{criteria.name}_{conf.get_conf_tag()}'

        retarget_image(input_image_path, criteria, conf, outputs_dir)

if __name__ == '__main__':
    resample()

