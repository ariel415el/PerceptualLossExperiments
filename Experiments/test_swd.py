import os

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt

from losses.swd.swd import compute_swd


def test_swd_on_degraded_images(output_dir):
    def cv2pt(img):
        img = img.astype(np.float64) / 255.
        img = img * 2 - 1
        img = torch.from_numpy(img.transpose(2, 0, 1)).float()
        return img
    os.makedirs(output_dir, exist_ok=True)
    x = '/home/ariel/university/PerceptualLoss/sty  ',
    paths = [
        # '/home/ariel/university/PerceptualLoss/PerceptualLossExperiments/style_transfer/imgs/content/bair.jpg',
        # '/home/ariel/university/PerceptualLoss/PerceptualLossExperiments/style_transfer/imgs/content/modern.jpg',
        # '/home/ariel/university/PerceptualLoss/PerceptualLossExperiments/style_transfer/imgs/content/lenna.jpg',
        '/home/ariel/university/PerceptualLoss/PerceptualLossExperiments/style_transfer/imgs/content/green_eye.jpg',
        # '/home/ariel/university/PerceptualLoss/PerceptualLossExperiments/style_transfer/imgs/faces/FFHQ-0.png'
    ]
    for path in paths:
        img_np = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        img_np = cv2.resize(img_np, (128, 128))

        def noisy(x, i):
            return np.clip(x + np.random.normal(0, 10 * (i+1), x.shape), 0, 255), f"+ N(0,{10 * (i+1)})"

        def blur(x, i):
            return cv2.blur(x, ksize=(2 + i, 2 + i)), f"Blur(k={2 +  i})"

        def compress(x, i):
            return cv2.imdecode(cv2.imencode('.jpg', x, [int(cv2.IMWRITE_JPEG_QUALITY), 90 - i * 17])[1], 1), f"JPG({90 - i * 17})"

        degrade_funcs = [noisy, blur, compress]

        n = 6
        font_size = 12
        fig, axs = plt.subplots(len(degrade_funcs) + 1, n, figsize=(20, 10))#, gridspec_kw={'wspace': 0, 'hspace': 0.25})

        axs[0, 0].imshow(img_np)
        axs[0, 0].set_title(f"orignal", size=font_size)
        axs[0, 0].axis('off')

        img_pt = cv2pt(img_np)[None, :]
        for i in range(n):
            for j, deg_func in enumerate(degrade_funcs):
                degraded_img, name = deg_func(img_np, i)
                swd_score = compute_swd(img_pt, cv2pt(degraded_img)[None, :], device="cuda")
                axs[j + 1, i].imshow(degraded_img.astype(int))
                axs[j + 1, i].set_title(f"{name}\nSWD:{swd_score:.1f}", size=font_size)
                axs[j + 1, i].axis('off')
                axs[j + 1, i].set_aspect('equal')

            if i > 0:
                fig.delaxes(axs[0, i])

        plt.tight_layout()
        img_name = os.path.splitext(os.path.basename(path))
        plt.savefig(f"{output_dir}/SWD-{img_name}.png")


if __name__ == '__main__':
    device = torch.device("cuda")
    outputs_dir = 'Experiments/test_swd'
    test_swd_on_degraded_images(outputs_dir)
