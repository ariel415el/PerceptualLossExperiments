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


def generate_texture():
    for texture_image_path in [
        'images/textures/olives.png',
        'images/textures/tomatos.png',
        'images/textures/green_waves.jpg',
        'images/textures/cobbles.jpeg'
    ]:
        # criteria = losses.MMDApproximate(patch_size=11, strides=1, sigma=0.04, pool_size=-1, r=256, normalize_patch='none')
        criteria = losses.PatchSWDLoss(patch_size=7, stride=1, num_proj=512, normalize_patch='none')

        conf = SyntesisConfigurations(pyr_factor=0.6, n_scales=5, aspect_ratio=(2,2), lr=0.01, num_steps=500, init="noise", resize=256)

        outputs_dir = f'outputs/texture_synthesis/{get_file_name(texture_image_path)}/{criteria.name}_{conf.get_conf_tag()}'

        retarget_image(texture_image_path, criteria, conf, outputs_dir)


def style_transfer():
    for style_image_path, content_image_path in [
        ('images/analogies/duck_mosaic.jpg', 'images/analogies/S_char.jpg'),
        ('images/analogies/S_char.jpg', 'images/analogies/duck_mosaic.jpg'),
        ('/home/ariel/university/PerceptualLoss/PerceptualLossExperiments/style_transfer/imgs/style/scream.jpg', '/home/ariel/university/PerceptualLoss/PerceptualLossExperiments/style_transfer/imgs/content/home_alone.jpg'),
        ('/home/ariel/university/PerceptualLoss/PerceptualLossExperiments/style_transfer/imgs/style/yellow_sunset.jpg', '/home/ariel/university/PerceptualLoss/PerceptualLossExperiments/style_transfer/imgs/content/bair.jpg')
    ]:
        criteria = losses.PatchSWDLoss(patch_size=11, stride=1, num_proj=256, normalize_patch='channel_mean')

        conf = SyntesisConfigurations(n_scales=0, lr=0.05, num_steps=1000, init=content_image_path, resize=256, tv_loss=0.1)

        outputs_dir = f'outputs/style_transfer/{get_file_name(content_image_path)}-to-{get_file_name(style_image_path)}/{criteria.name}_{conf.get_conf_tag()}'

        retarget_image(style_image_path, criteria, conf, outputs_dir)


def edit_image():
    for style_image_path, content_image_path in [
        ('images/resampling/balls.jpg', 'images/edit_inputs/balls_green_and_black.jpg'),
        ('images/resampling/birds.png', 'images/edit_inputs/birds_edit_1.jpg'),
        ('images/resampling/balloons.png', 'images/edit_inputs/balloons_edit.jpg'),
        ('images/image_editing/stone.png', 'images/edit_inputs/stone_edit.png'),
        ('images/image_editing/tree.png', 'images/edit_inputs/tree_edit.png'),
        ('images/image_editing/swiming1.jpg', 'images/edit_inputs/swiming1_edit.jpg'),
    ]:
        for (patch_size, n_scales) in [(11,0)]:
            criteria = losses.PatchSWDLoss(patch_size=patch_size, stride=1, num_proj=512, normalize_patch='mean')

            conf = SyntesisConfigurations(n_scales=n_scales, pyr_factor=0.65, lr=0.05, num_steps=500, init=content_image_path, resize=256, tv_loss=0)

            outputs_dir = f'outputs/image_editing/{get_file_name(content_image_path)}-to-{get_file_name(style_image_path)}/{criteria.name}_{conf.get_conf_tag()}'

            retarget_image(style_image_path, criteria, conf, outputs_dir)


def image_resampling():
    """
    Smaller patch size adds variablility but may ruin large objects
    """
    for input_image_path in [
        # 'images/resampling/cows.png',
        # 'images/resampling/balloons.png',
        # 'images/resampling/people_on_the_beach.jpg',
        # 'images/resampling/balls.jpg',
        # 'images/resampling/green_view.jpg',
        # 'images/resampling/birds.png',
        # 'images/resampling/jerusalem2.jpg',
    ]:
        for i in range(3):
            # criteria = losses.MMDApproximate(patch_size=5, strides=1, sigma=0.03, pool_size=-1, r=512, normalize_patch='channel_mean')
            criteria = losses.PatchSWDLoss(patch_size=7, stride=1, num_proj=256, normalize_patch='none')

            conf = SyntesisConfigurations(pyr_factor=0.75, n_scales=5, lr=0.05, num_steps=1000, init="noise", resize=256)

            outputs_dir = f'outputs/image_resampling/{get_file_name(input_image_path)}/{criteria.name}_{conf.get_conf_tag()}'

            retarget_image(input_image_path, criteria, conf, outputs_dir)


def image_retargeting():
    for input_image_path in [
        # 'images/retargeting/mountains3.png',
        # 'images/retargeting/birds2.jpg',
        # 'images/retargeting/fish.png',
        # 'images/retargeting/fruit.png',
        # 'images/retargeting/mountins2.jpg',
        # 'images/retargeting/colusseum.png',
        # 'images/retargeting/mountains.jpg',
        'images/retargeting/SupremeCourt.jpeg',
        'images/retargeting/kanyon.jpg',
        'images/retargeting/corn.png',
        'images/resampling/balloons.png',
        'images/resampling/birds.png'
        'images/resampling/pinguins.png',
        'images/resampling/birds.png',
    ]:
        for aspect_ratio in [(1,0.5), (1, 1.5)]:
            # criteria = losses.MMDApproximate(patch_size=11, strides=1, sigma=0.03, pool_size=-1, r=512, normalize_patch='channel_mean')
            criteria = losses.PatchSWDLoss(patch_size=5, stride=1, num_proj=1024, normalize_patch='channel_mean')

            conf = SyntesisConfigurations(pyr_factor=0.6, n_scales=5, aspect_ratio=aspect_ratio, lr=0.05, num_steps=500, init="blur", resize=256)

            outputs_dir = f'outputs/image_retargeting/{get_file_name(input_image_path)}_AR-{conf.aspect_ratio}/{criteria.name}_{conf.get_conf_tag()}'

            retarget_image(input_image_path, criteria, conf, outputs_dir)


if __name__ == '__main__':
    generate_texture()
    style_transfer()
    edit_image()
    # image_resampling()
    image_retargeting()

