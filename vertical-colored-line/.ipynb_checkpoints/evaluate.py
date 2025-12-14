import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

img1 = cv2.imread('./debug/raw_frame_30.png', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('./debug/denoise_frame_30.png', cv2.IMREAD_GRAYSCALE)

def mse(x, y):
    return np.mean((x - y) ** 2)

# --- PSNR ---
def psnr(x, y):
    mse_val = mse(x, y)
    if mse_val == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse_val))

# --- SSIM ---
ssim_val, ssim_map = ssim(img1, img2, full=True)

# --- DSSIM ---
dssim_val = (1 - ssim_val) / 2

print("MSE:", mse(img1, img2))
print("PSNR:", psnr(img1, img2))
print("SSIM:", ssim_val)
print("DSSIM:", dssim_val)
