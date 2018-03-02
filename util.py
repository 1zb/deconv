import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def read_grey_image(filename):
    image = np.array(Image.open(filename).convert('L')) / 255.0
    return image


def to_pil(image):
    return Image.fromarray(np.uint8(image * 255))


def gen_gaussian_kernel(width=3, sig=1):
    ax = np.arange(-width, width + 1.)
    xx, yy = np.meshgrid(ax, ax)

    kernel = np.exp(-(xx**2 + yy**2) / (2. * sig**2))

    return kernel / np.sum(kernel)


def plot_cost_vs_time(loss, time):
    t = np.cumsum(time)
    plt.plot(t, loss, 'x')
    plt.xlabel('Time (sec)')
    plt.ylabel('Cost Functional')
    plt.show()


def plot_multiple_cost_vs_time(losses, times, labels, title, sigma, lmbda):
    fig, axs = plt.subplots(3, 2, figsize=(14, 7), sharex=True)

    fig.text(0.5, 0.04, 'Time (sec)', ha='center')
    fig.text(0.04, 0.5, 'Cost Functional', va='center', rotation='vertical')

    for i in range(len(axs.flatten())):
        for loss, time, label in zip(losses, times, labels):
            t = np.cumsum(time[i])
            axs.flatten()[i].plot(t, loss[i], 'x-', label=label)
            axs.flatten()[i].set_title(title.format(sigma[i], lmbda[i]))

    plt.legend()
    plt.show()


def plot_multiple_metric_vs_sigma(metrics,
                                  noise_levels,
                                  labels,
                                  y='PSNR',
                                  title=None):
    plt.figure(figsize=(14, 7))
    for metric, label in zip(metrics, labels):
        # t = np.cumsum(time)
        plt.plot(noise_levels, metric, 'x-', label=label)
    plt.xlabel('Sigma')
    plt.ylabel(y)
    if title is not None:
        plt.title(title)
    plt.legend()

    plt.show()
