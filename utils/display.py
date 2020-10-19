import io

import matplotlib.pyplot as plt
import numpy as np


def buffer_image(figure):
    buf = io.BytesIO()
    figure.savefig(buf, format='png')
    buf.seek(0)
    plt.close('all')
    return buf


def gen_plot(image, with_bar, figsize=None, title=''):
    """Create a pyplot plot and save to buffer."""
    f = plt.figure(figsize=figsize)
    plt.imshow(image)
    plt.title(title)
    if with_bar:
        plt.colorbar()
    buf = buffer_image(f)
    return buf


def tight_grid(images):
    images = np.array(images)
    images = np.pad(images, [[0, 0], [1, 1], [1, 1]], 'constant', constant_values=1)  # add borders
    if len(images.shape) != 3:
        raise Exception
    else:
        n, y, x = images.shape
    ratio = y / x
    if ratio > 1:
        ny = max(int(np.sqrt(n / ratio)), 1)
        nx = int(n / ny)
        nx += n - (nx * ny)
        extra = nx * ny - n
    else:
        nx = max(int(np.sqrt(n * ratio)), 1)
        ny = int(n / nx)
        ny += n - (nx * ny)
        extra = nx * ny - n
    tot = np.append(images, np.zeros((extra, y, x)), axis=0)
    img = np.block([[*tot[i * nx:(i + 1) * nx]] for i in range(ny)])
    return img
