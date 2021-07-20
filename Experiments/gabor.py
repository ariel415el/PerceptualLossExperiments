import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import gabor_kernel


def gabor(sigma, theta, Lambda, psi, gamma):
    """Gabor feature extraction."""
    sigma_x = float(sigma) / gamma
    sigma_y = float(sigma) / gamma

    # Bounding box
    nstds = 3  # Number of standard deviation sigma
    xmax = max(abs(nstds * sigma_x * np.cos(theta)), abs(nstds * sigma_y * np.sin(theta)))
    xmax = np.ceil(max(1, xmax))
    ymax = max(abs(nstds * sigma_x * np.sin(theta)), abs(nstds * sigma_y * np.cos(theta)))
    ymax = np.ceil(max(1, ymax))
    xmin = -xmax
    ymin = -ymax
    (y, x) = np.meshgrid(np.arange(ymin, ymax + 1), np.arange(xmin, xmax + 1))

    # Rotation
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)

    gb = np.exp(-.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2)) * np.cos(
        2 * np.pi / Lambda * x_theta + psi)
    return gb


if __name__ == '__main__':
    fif, axis = plt.subplots(10, 5)
    for i in range(5):
        for j in range(5):
            gb = gabor_kernel(0.1 + 0.05 * i, theta=0.45, sigma_x=0.1 + 0.1*j, sigma_y=0.1 + 0.1*j)
            # gb = gabor(1 + i*5, 0.45, 1, 1, 1)
            axis[2 * i, j].imshow(gb.real)
            axis[2 * i, j].set_title(f"freq: {0.1 + 0.05 * i:.2f} sigma: {1 + j}", size=6)
            axis[2 * i, j].set_axis_off()
            axis[2 * i + 1, j].imshow(gb.imag)
            axis[2 * i, j].set_title(gb.shape, size=6)
            axis[2 * i+1, j].set_axis_off()
    plt.show()
