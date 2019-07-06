""" TensorMONK :: utils """

import torch
from torch import Tensor
import numpy as np
from skimage.filters import gabor_kernel


class Kernels(object):

    @staticmethod
    def gaussian(sigma: float, width: int, normalize: bool = True) -> Tensor:
        if width is None:
            width = int(2.0 * 3.0 * sigma + 1.0)
        if width % 2 == 0:
            width += 1
        if sigma is None or sigma == 0:
            sigma = (width - 1)/6.
        half = width // 2
        x, y = np.meshgrid(np.linspace(-half, half, width),
                           np.linspace(-half, half, width), indexing="xy")
        w = np.exp(- (x**2 + y**2) / (2.*(sigma**2))).astype(np.float32)
        if normalize:
            w /= np.sum(w)
        return torch.from_numpy(w).view(1, 1, width, width)

    @staticmethod
    def log(sigma: float, width: int, normalize: bool = True) -> Tensor:
        width, pad = (width, True) if width is None else (width+4, False)
        gaussian = Kernels.gaussian(sigma, width, normalize)
        return Kernels.laplacian(gaussian, pad)

    @staticmethod
    def gabor(sigma: float, wave_length: float, theta: float, gamma: float,
              n_stds: float = 3.) -> Tensor:
        sigma_x = sigma
        sigma_y = float(sigma) / gamma
        kernels = gabor_kernel(frequency=1/wave_length, theta=theta,
                               sigma_x=sigma_x, sigma_y=sigma_y,
                               n_stds=n_stds)
        gab_real = np.real(kernels).astype(np.float32)
        gab_imag = np.imag(kernels).astype(np.float32)
        gab_real = gab_real.reshape(1, 1, *gab_real.shape)
        gab_imag = gab_imag.reshape(1, 1, *gab_imag.shape)
        return torch.cat((torch.from_numpy(gab_real),
                          torch.from_numpy(gab_imag)))

    @staticmethod
    def laplacian(tensor: Tensor, pad: bool) -> Tensor:
        if pad:
            tensor = torch.nn.functional.pad(tensor, (2, 2, 2, 2))
        gx = tensor[:, :, 1:-1, 2:] - tensor[:, :, 1:-1, :-2]
        gy = tensor[:, :, 2:, 1:-1] - tensor[:, :, :-2, 1:-1]

        gxx = gx[:, :, 1:-1, 2:] - gx[:, :, 1:-1, :-2]
        gyy = gy[:, :, 2:, 1:-1] - gy[:, :, :-2, 1:-1]
        return gxx + gyy

    @staticmethod
    def sobel(normalize: bool = True) -> Tensor:
        kernel = Tensor([[[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
                         [[1, 0, -1], [2, 0, -2], [1, 0, -1]]])
        kernel.unsqueeze_(1)
        if normalize:
            l2 = kernel.pow(2).sum(2, True).sum(3, True).pow(0.5)
            kernel.div_(l2)
        return kernel.float()

    @staticmethod
    def anisotropic_gaussian(sigma_x: float, sigma_y: float, theta: float,
                             width: int = None) -> Tensor:
        if width is None:
            width = int(max(2. * 3. * sigma_x + 1., 2. * 3. * sigma_y + 1.))
        if width % 2 == 0:
            width += 1
        half = width//2
        x, y = np.meshgrid(np.linspace(-half, half, width),
                           np.linspace(-half, half, width), indexing="xy")

        x_theta = x * np.cos(theta) + y * np.sin(theta)
        y_theta = -x * np.sin(theta) + y * np.cos(theta)

        kernel = ((x_theta**2) / (sigma_x**2)) + ((y_theta**2) / (sigma_y**2))
        kernel = np.exp(- 0.5 * kernel).astype(np.float32)
        return torch.from_numpy(kernel).view(1, 1, width, width)

    @staticmethod
    def anisotropic_log(sigma_x: float, sigma_y: float, theta: float,
                        width: int) -> Tensor:
        if width % 2 == 0:
            width += 1
        gaussian = Kernels.anisotropic_gaussian(sigma_x, sigma_y, theta,
                                                width+4)
        return Kernels.laplacian(gaussian, pad=False)

    @staticmethod
    def top_n_kernels(kernels: Tensor, n_kernels: int,
                      method: str = "kpca") -> Tensor:
        n, c, h, w = kernels.shape

        if method.lower() == "kpca":
            from sklearn.decomposition import KernelPCA
            transformer = KernelPCA(n_components=n_kernels, kernel="rbf")
            kernels = transformer.fit_transform(
                kernels.view(-1, h*w).numpy().T)
            kernels = torch.from_numpy((kernels.T)).float().view(-1, 1, h, w)
        elif method.lower() == "svd":
            from sklearn.decomposition import TruncatedSVD
            svd = TruncatedSVD(n_components=n_kernels, n_iter=646,
                               random_state=46)
            svd.fit(kernels.view(-1, h*w).numpy())

            kernels = svd.components_
            kernels *= (svd.singular_values_ *
                        svd.explained_variance_).reshape(-1, 1)
            kernels = kernels.sum(0)
            kernels = torch.from_numpy(kernels).float().view(1, 1, h, w)
        else:
            raise NotImplementedError
        return kernels
