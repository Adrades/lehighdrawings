import os
import argparse
import cv2
import numpy as np

from numba import njit, prange, float64, uint8


# imageName = 'images/image.tif'
binaryThreshold = 5
histogramThreshold = 0.96



class MaskerParser(argparse.ArgumentParser):
    def __init__(self):
        super().__init__(
            prog="Masker",
            description="Create mask for images of the lehighdrawings dataset",
            epilog="See in readme.md and pyproject.toml for more informations",
        )
        self.define_args()

    def define_args(self):
        self.add_argument(
            "-i",
            "--input",
            default=[],
            action="append",
            help="Add an image to mask",
        )

        self.add_argument(
            "-o",
            "--output",
            default="./",
            action="store",
            type=str,
            help="Folder where mask are saved",
        )

        self.add_argument(
            "-v", "--verbose", action="store", help="Print more logging messages"
        )


class Masker:
    def __init__(self):
        args = MaskerParser().parse_args()
        self.folder = args.output
        self.images = args.input

        for img in self.images:
            name = os.path.basename(img)
            self.create_mask(img, name)

    def create_mask(self, img_path, img_name):
        # Load image
        img = cv2.imread(img_path)

        # Create float
        # Extract channels
        bgr = get_bgr(img)
        K = extract_K(bgr)
        C = extract_channels(bgr, K, 2)
        M = extract_channels(bgr, K, 1)
        Y = extract_channels(bgr, K, 0)

        np.isfinite(C).all()
        np.isfinite(M).all()
        np.isfinite(Y).all()

        _, C = cv2.threshold(C, binaryThreshold, 255, cv2.THRESH_BINARY_INV)
        _, M = cv2.threshold(M, binaryThreshold, 255, cv2.THRESH_BINARY_INV)
        _, Y = cv2.threshold(Y, binaryThreshold, 255, cv2.THRESH_BINARY_INV)

        res1 = cv2.bitwise_and(C, M)
        res2 = cv2.bitwise_and(C, Y)
        res3 = cv2.bitwise_and(M, Y)

        res = cv2.bitwise_or(C, M)
        res = cv2.bitwise_or(res, Y)

        res = res - res1 - res2 - res3
        _, res = cv2.threshold(res, 254, 255, cv2.THRESH_BINARY)

        # cv2.imwrite('test_tot.tif', res)

        histogram = np.sum(res, axis=0)

        res[:, histogram > 0] = 255
        res[:, histogram < res.shape[0] * 255 * histogramThreshold] = 0

        cv2.imwrite(f"{self.folder}/mask_{img_name}", res)


@njit(float64[:,:,:](uint8[:,:,:]), parallel=True)
def get_bgr(img):
    bgr = np.zeros(img.shape)
    for i in prange(bgr.shape[0]):
        for j in prange(bgr.shape[1]):
            for k in prange(bgr.shape[2]):
                bgr[i, j, k] = np.float64(img[i, j, k]) / 255
    # bgr = np.array(img, dtype=np.float64) / 255.
    return bgr

@njit(float64[:,:](float64[:,:,:]), parallel=True)
def extract_K(bgr):
    K = np.zeros((bgr.shape[0], bgr.shape[1]))

    #with np.errstate(invalid='ignore', divide='ignore'):
    # K = np.max(bgr, axis=2)
    for i in prange(bgr.shape[0]):
        for j in prange(bgr.shape[1]):
            K[i, j] = bgr[i, j].max()
    return K

@njit(uint8[:,:,:](float64[:,:,:], float64[:,:], uint8), parallel=True)
def extract_channels(bgr, K, i_chan):
    chan = np.zeros(bgr.shape, dtype = np.uint8)
    for i in prange(bgr.shape[0]):
        for j in prange(bgr.shape[1]):
            ki = K[i, j]
            chan[i, j] = np.uint8((1 - bgr[i, j, i_chan] / ki ) * 255)
    return chan

if __name__ == "__main__":
    Masker()

