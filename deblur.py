import time

import numpy as np
from scipy.ndimage import convolve, correlate
from scipy.sparse.linalg import aslinearoperator, cg
from skimage.measure import compare_psnr, compare_ssim


## define linear operator P * x
class Blur():
    def __init__(self, kernel, spatial_size):
        self.kernel = kernel
        self.shape = (spatial_size[0] * spatial_size[1],
                      spatial_size[0] * spatial_size[1])
        self.spatial_size = spatial_size

    def matvec(self, x):
        # conv(x, kernel)
        return convolve(x.reshape(self.spatial_size), self.kernel).flatten()


def deblur(kernel, blurred):
    image_size = blurred.shape

    A = aslinearoperator(Blur(kernel, image_size))
    b = blurred.copy()
    x, _ = cg(A, b.flatten(), maxiter=200)
    x.resize(image_size)
    return x.clip(0, 1)


## define linear operator P^T * P * x + mu * x
class BlurNormal():
    def __init__(self, kernel, spatial_size, mu=0):
        self.kernel = kernel
        self.shape = (spatial_size[0] * spatial_size[1],
                      spatial_size[0] * spatial_size[1])
        self.spatial_size = spatial_size
        self.mu = mu

    def matvec(self, x):
        return correlate(
            convolve(x.reshape(self.spatial_size), self.kernel),
            self.kernel).flatten() + self.mu * x


def deblur_normal(kernel, blurred, mu=0):
    image_size = blurred.shape

    A = aslinearoperator(BlurNormal(kernel, image_size, mu=mu))
    b = correlate(blurred, kernel)
    x, _ = cg(A, b.flatten(), maxiter=200)
    x.resize(image_size)
    return x.clip(0, 1)


## define linear operator P^T * P * x + lambda * D^T * W * D * x
class ITVReweight():
    def __init__(self,
                 kernel,
                 spatial_size,
                 weight,
                 lmbda=1e-5,
                 kernel_h=[[-1, 1]],
                 kernel_v=[[-1], [1]]):
        self.kernel = kernel
        self.shape = (spatial_size[0] * spatial_size[1],
                      spatial_size[0] * spatial_size[1])
        self.spatial_size = spatial_size
        self.weight = weight
        self.lmbda = lmbda
        self.kernel_h = kernel_h
        self.kernel_v = kernel_v

    def matvec(self, x):
        x = x.copy()
        x.resize(self.spatial_size)

        Dh = convolve(x, self.kernel_h)
        Dv = convolve(x, self.kernel_v)

        W_Dh = np.multiply(self.weight, Dh)
        Dht_W_Dh = correlate(W_Dh, self.kernel_h)

        W_Dv = np.multiply(self.weight, Dv)
        Dvt_W_Dv = correlate(W_Dv, self.kernel_v)

        return (correlate(convolve(x, self.kernel), self.kernel) + self.lmbda *
                (Dht_W_Dh + Dvt_W_Dv)).flatten()


def deblur_itv_reweight(kernel,
                        blurred,
                        p=1,
                        lmbda=1e-5,
                        irls_iters=10,
                        cg_iters=100,
                        groundtruth=None,
                        verbose=False):
    image_size = blurred.shape
    kernel_h = [[-1, 1]]
    kernel_v = [[-1], [1]]

    b = correlate(blurred, kernel)
    # W = np.ones(image_size)
    x = blurred.copy()

    t = []
    loss = []

    for irls_iter in range(irls_iters):
        x_old = x

        t0 = time.time()

        Dh_x = convolve(x, kernel_h)
        Dv_x = convolve(x, kernel_v)
        W = np.power(
            np.maximum(np.power(Dh_x, 2) + np.power(Dv_x, 2), 1e-4), p / 2 - 1)

        A = aslinearoperator(
            ITVReweight(
                kernel,
                image_size,
                W,
                lmbda=lmbda,
                kernel_h=kernel_h,
                kernel_v=kernel_v))
        x, _ = cg(A, b.flatten(), maxiter=cg_iters)
        x = x.clip(0, 1)
        x.resize(image_size)

        t1 = time.time()

        obj = 0.5 * np.sum(np.power(
            convolve(x, kernel) - blurred, 2)) + lmbda * (np.sum(
                np.sqrt(
                    np.power(convolve(x, kernel_h), 2) +
                    np.power(convolve(x, kernel_v), 2))))

        if verbose:
            if groundtruth is not None:
                print(
                    'IRLS Iter: {}; Cost Funcitonal: {:.6f}; PSNR: {:.6f}; SSIM: {:.6f}'.
                    format(irls_iter + 1, obj, compare_psnr(x, groundtruth),
                           compare_ssim(x, groundtruth)))
            else:
                print('IRLS Iter: {}; Cost Funcitonal: {:.6f};'.format(
                    irls_iter + 1, obj))

        t.append(t1 - t0)
        loss.append(obj)

        if np.sum(np.power(x_old - x, 2)) < 1e-2:
            break

    return x, t, loss


## define linear operator P^T * P * x + lambda * D^T * W * D * x
class ATVReweight():
    def __init__(self,
                 kernel,
                 spatial_size,
                 weight_h,
                 weight_v,
                 lmbda=1e-5,
                 kernel_h=[[-1, 1]],
                 kernel_v=[[-1], [1]]):
        self.kernel = kernel
        self.shape = (spatial_size[0] * spatial_size[1],
                      spatial_size[0] * spatial_size[1])
        self.spatial_size = spatial_size
        self.weight_h = weight_h
        self.weight_v = weight_v
        self.lmbda = lmbda
        self.kernel_h = kernel_h
        self.kernel_v = kernel_v

    def matvec(self, x):
        x = x.copy()
        x.resize(self.spatial_size)

        Dh = convolve(x, self.kernel_h)
        Dv = convolve(x, self.kernel_v)

        W_Dh = np.multiply(self.weight_h, Dh)
        Dht_W_Dh = correlate(W_Dh, self.kernel_h)

        W_Dv = np.multiply(self.weight_v, Dv)
        Dvt_W_Dv = correlate(W_Dv, self.kernel_v)

        return (correlate(convolve(x, self.kernel), self.kernel) + self.lmbda *
                (Dht_W_Dh + Dvt_W_Dv)).flatten()


def deblur_atv_reweight(kernel,
                        blurred,
                        lmbda=1e-5,
                        irls_iters=10,
                        cg_iters=100,
                        groundtruth=None,
                        verbose=False):
    image_size = blurred.shape
    kernel_h = [[-1, 1]]
    kernel_v = [[-1], [1]]

    b = correlate(blurred, kernel)
    # W = np.ones(image_size)
    x = blurred.copy()

    t = []
    loss = []

    for irls_iter in range(irls_iters):
        x_old = x

        t0 = time.time()

        Dh_x = convolve(x, kernel_h)
        Dv_x = convolve(x, kernel_v)
        Wh = 1 / np.maximum(np.abs(Dh_x), 1e-4)
        Wv = 1 / np.maximum(np.abs(Dv_x), 1e-4)

        A = aslinearoperator(
            ATVReweight(
                kernel,
                image_size,
                Wh,
                Wv,
                lmbda=lmbda,
                kernel_h=kernel_h,
                kernel_v=kernel_v))

        x, _ = cg(A, b.flatten(), maxiter=cg_iters)
        x = x.clip(0, 1)
        x.resize(image_size)

        t1 = time.time()

        obj = 0.5 * np.sum(np.power(convolve(x, kernel) - blurred, 2)
                           ) + lmbda * (np.sum(np.abs(convolve(x, kernel_h))) +
                                        np.sum(np.abs(convolve(x, kernel_v))))

        if verbose:
            if groundtruth is not None:
                print(
                    'IRLS Iter: {}; Cost Funcitonal: {:.6f}; PSNR: {:.6f}; SSIM: {:.6f}'.
                    format(irls_iter + 1, obj, compare_psnr(x, groundtruth),
                           compare_ssim(x, groundtruth)))
            else:
                print('IRLS Iter: {}; Cost Funcitonal: {:.6f};'.format(
                    irls_iter + 1, obj))

        t.append(t1 - t0)
        loss.append(obj)

        if np.sum(np.power(x_old - x, 2)) < 1e-2:
            break

    return x, t, loss


## define linear operator P^T * P * x + mu * x
class ADMMLinearize():
    def __init__(self, kernel, spatial_size, mu):
        self.kernel = kernel
        self.shape = (spatial_size[0] * spatial_size[1],
                      spatial_size[0] * spatial_size[1])
        self.spatial_size = spatial_size
        self.mu = mu

    def matvec(self, x):
        # the linear operator is (P^T * P + mu * I)
        return correlate(
            convolve(x.reshape(self.spatial_size), self.kernel),
            self.kernel).flatten() + self.mu * x


def deblur_admm_linearized(kernel,
                           blurred,
                           lmbda=1e-5,
                           mu=1e-4,
                           rho=1e-4,
                           admm_iters=10,
                           cg_iters=100,
                           groundtruth=None):
    image_size = blurred.shape
    kernel_h = [[-1, 1]]
    kernel_v = [[-1], [1]]

    def update_x(x, z, y, rho, mu):
        b = correlate(blurred, kernel)  # P^T * b
        b += mu * x  # mu * x

        ## rho * D * x - rho * z + y
        # horizontal part
        b -= correlate(rho * convolve(x, kernel_h) -
                       rho * z[:z.shape[0] // 2, :] + y[:y.shape[0] // 2, :],
                       kernel_h)
        # vertitcal part
        b -= correlate(rho * convolve(x, kernel_v) -
                       rho * z[z.shape[0] // 2:, :] + y[y.shape[0] // 2:, :],
                       kernel_v)

        A = aslinearoperator(ADMM(kernel, image_size, mu))

        x, _ = cg(A, b.flatten(), maxiter=cg_iters)
        x.resize(image_size)
        return x

    def update_z(x, y, lmbda, rho):
        z = np.vstack([convolve(x, kernel_h), convolve(x, kernel_v)]) + y / rho
        threshold = lmbda / rho
        j = np.abs(z) <= threshold
        z[j] = 0
        k = np.abs(z) > threshold
        z[k] = z[k] - np.sign(z[k]) * threshold
        return z

    def update_y(x, z, y, rho):
        return y + rho * (
            np.vstack([convolve(x, kernel_h),
                       convolve(x, kernel_v)]) - z)

    x = blurred.copy()
    # x = np.zeros(blurred.shape)
    z = np.vstack([np.zeros(x.shape), np.zeros(x.shape)])
    y = np.vstack([np.zeros(x.shape), np.zeros(x.shape)])

    t = []
    loss = []
    for admm_iter in range(admm_iters):
        x_old = x

        t0 = time.time()

        x = update_x(x, z, y, rho, mu)
        x = x.clip(0, 1)
        z = update_z(x, y, lmbda, rho)
        y = update_y(x, z, y, rho)
        rho = min(5, rho * 1.1)
        mu = min(5, mu * 1.1)

        t1 = time.time()

        obj = 0.5 * np.sum(np.power(convolve(x, kernel) - blurred, 2)
                           ) + lmbda * (np.sum(np.abs(convolve(x, kernel_h))) +
                                        np.sum(np.abs(convolve(x, kernel_v))))

        if groundtruth is not None:
            print(
                'ADMM Iter: {}; Cost Functional: {:.6f}; PSNR: {:.6f}; SSIM: {:.6f}'.
                format(admm_iter + 1, obj, compare_psnr(x, groundtruth),
                       compare_ssim(x, groundtruth)))
        else:
            print('ADMM Iter: {}; Cost Functional: {:.6f}; '.format(
                admm_iter + 1, obj))

        t.append(t1 - t0)
        loss.append(obj)

        if np.sum(np.power(x_old - x, 2)) < 1e-2:
            break

    return x, t, loss


## define linear operator P^T * P * x + mu * D^T * D * x
class ADMM():
    def __init__(self, kernel, spatial_size, kernel_h, kernel_v, rho):
        self.kernel = kernel
        self.shape = (spatial_size[0] * spatial_size[1],
                      spatial_size[0] * spatial_size[1])
        self.spatial_size = spatial_size
        self.rho = rho
        self.kernel_h = kernel_h
        self.kernel_v = kernel_v

    def matvec(self, x):
        # the linear operator is P^T * P * x + mu * D^T * D * x
        x = x.copy()
        x.resize(self.spatial_size)

        return correlate(convolve(x, self.kernel), self.kernel).flatten(
        ) + self.rho * correlate(convolve(
            x, self.kernel_h), self.kernel_h).flatten() + self.rho * correlate(
                convolve(x, self.kernel_v), self.kernel_v).flatten()


def deblur_admm(kernel,
                blurred,
                lmbda=1e-5,
                rho=1e-4,
                admm_iters=25,
                cg_iters=100,
                groundtruth=None,
                verbose=False):
    image_size = blurred.shape
    kernel_h = [[-1, 1]]
    kernel_v = [[-1], [1]]

    def update_x(z, y, rho):
        b = correlate(blurred, kernel)  # P^T * b

        ## D^T * (rho * z - y)

        # horizontal part
        b += correlate(rho * z[:z.shape[0] // 2, :] - y[:y.shape[0] // 2, :],
                       kernel_h)
        # vertitcal part
        b += correlate(rho * z[z.shape[0] // 2:, :] - y[y.shape[0] // 2:, :],
                       kernel_v)

        A = aslinearoperator(ADMM(kernel, image_size, kernel_h, kernel_v, rho))

        x, _ = cg(A, b.flatten(), maxiter=cg_iters)
        x.resize(image_size)
        return x

    def update_z(x, y, lmbda, rho):
        z = np.vstack([convolve(x, kernel_h), convolve(x, kernel_v)]) + y / rho
        threshold = lmbda / rho
        j = np.abs(z) <= threshold
        z[j] = 0
        k = np.abs(z) > threshold
        z[k] = z[k] - np.sign(z[k]) * threshold
        return z

    def update_y(x, z, y, rho):
        return y + rho * (
            np.vstack([convolve(x, kernel_h),
                       convolve(x, kernel_v)]) - z)

    x = blurred.copy()
    # x = np.zeros(blurred.shape)
    z = np.vstack([np.zeros(x.shape), np.zeros(x.shape)])
    y = np.vstack([np.zeros(x.shape), np.zeros(x.shape)])

    t = []
    loss = []
    for admm_iter in range(admm_iters):
        x_old = x

        t0 = time.time()

        x = update_x(z, y, rho)
        x = x.clip(0, 1)
        z = update_z(x, y, lmbda, rho)
        y = update_y(x, z, y, rho)
        rho = min(5, rho * 1.1)

        t1 = time.time()

        obj = 0.5 * np.sum(np.power(convolve(x, kernel) - blurred, 2)
                           ) + lmbda * (np.sum(np.abs(convolve(x, kernel_h))) +
                                        np.sum(np.abs(convolve(x, kernel_v))))
        if verbose:
            if groundtruth is not None:
                print(
                    'ADMM Iter: {}; Cost Functional: {:.6f}; PSNR: {:.6f}; SSIM: {:.6f}'.
                    format(admm_iter + 1, obj, compare_psnr(x, groundtruth),
                           compare_ssim(x, groundtruth)))
            else:
                print('ADMM Iter: {}; Cost Functional: {:.6f};'.format(
                    admm_iter + 1, obj))

        t.append(t1 - t0)
        loss.append(obj)

        if np.sum(np.power(x_old - x, 2)) < 1e-2:
            break

    return x, t, loss
