import cv2
import numpy as np

imageName = 'images/image.tif'
binaryThreshold = 5
histogramThreshold = 0.96

# Load image
img = cv2.imread(imageName)

# Create float
bgr = img.astype(float) / 255.

# Extract channels
with np.errstate(invalid='ignore', divide='ignore'):
    K = np.max(bgr, axis=2)
    print('K')
    C = bgr[..., 2] / K
    C = 1 - C
    C = (C * 255).astype(np.uint8)
    print('C')
    M = bgr[..., 1] / K
    M = 1 - M
    M = (M * 255).astype(np.uint8)
    print('M')
    Y = bgr[..., 0] / K
    Y = 1 - Y
    Y = (Y * 255).astype(np.uint8)
    print('Y')

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

cv2.imwrite('mask.tif', res)
