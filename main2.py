# Image reconstruction and quality evaluation using cubic spline interpolation

from matplotlib import pyplot as plt
import numpy as np
import cv2
from PIL import Image
import src4 as src
import time
import pybundle
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# Load the input image and calibration image
img = np.array(Image.open("../Jiahui Shao/data/usaf1.tif"))
calibImg = np.array(Image.open("../Jiahui Shao/data/usaf1_background.tif"))

coreSize = 3
gridSize = 800

# Calibrate using cubic spline interpolation
t0 = time.time()
calib = src.calib_spline_interp(
    calibImg,
    coreSize,
    gridSize,
    background=None,
    normalise=calibImg,
    filterSize=0,
    mask=True,
    autoMask=True
)
print(f"   -> calib_spline_interp done in {time.time() - t0:.2f}s")

# Plot detected fibre cores
plt.figure(figsize=(8, 8))
plt.imshow(calibImg, cmap='gray')
plt.scatter(calib.coreX, calib.coreY, s=10, c='red', marker='o', label='Fibre cores')
plt.title("Detected Fibre Cores Overlay")
plt.axis('off')
plt.legend()
plt.show()

# Image reconstruction using cubic spline interpolation
t1 = time.time()
imgRecon = src.recon_spline_interp(img, calib, coreSize=coreSize)
print(f"   -> recon_spline_interp done in {time.time() - t1:.2f}s")

# Calibration and reconstruction using triangular interpolation (for comparison)
t0 = time.time()
calibSingle = pybundle.calib_tri_interp(
    calibImg, coreSize, gridSize,
    filterSize=0, normalise=calibImg,
    mask=True, autoMask=True
)
print(f"   -> calib_tri_interp done in {time.time() - t0:.2f}s")

t1 = time.time()
reconSingle = pybundle.recon_tri_interp(img, calibSingle)
print(f"   -> recon_tri_interp done in {time.time() - t1:.2f}s")

# Prepare the reconstructed image by padding into a canvas
image_clean = np.nan_to_num(imgRecon, nan=0)

h, w = image_clean.shape
canvas = np.full((800, 800), 0, dtype=image_clean.dtype)

start_y = (800 - h) // 2
start_x = (800 - w) // 2
canvas[start_y:start_y + h, start_x:start_x + w] = image_clean

# Visualization: reconstructed vs. original image
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image_clean, cmap='gray')
plt.axis('off')
plt.title('This research')

plt.subplot(1, 2, 2)
plt.imshow(img, cmap='gray')
plt.axis('off')
plt.title('Original image')
plt.tight_layout()
plt.show()

"""
# Visualization: this research vs. triangular interpolation
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image_clean, cmap='gray')
plt.axis('off')
plt.title("This research")

plt.subplot(1, 2, 2)
plt.imshow(reconSingle, cmap='gray')
plt.axis('off')
plt.title("Current latest research")
plt.tight_layout()
plt.show()
"""

# Function: Convert to grayscale if necessary
def to_gray(img):
    """Converts image to grayscale if it is RGB."""
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

# Convert images to grayscale
original_gray = to_gray(img)
tri_gray = to_gray(reconSingle)
spline_gray = to_gray(image_clean)

# Crop the original image to the center 800x800 region
h, w = original_gray.shape
crop_size = 800
start_x = (w - crop_size) // 2
start_y = (h - crop_size) // 2
original_crop = original_gray[start_y:start_y + crop_size, start_x:start_x + crop_size]

# Create a circular mask
mask = np.zeros((crop_size, crop_size), dtype=np.uint8)
cv2.circle(mask, (crop_size // 2, crop_size // 2), crop_size // 2, 1, thickness=-1)

# Function: Apply a binary mask to an image
def apply_mask(img, mask):
    """Applies a binary mask to an image."""
    return img * mask

# Apply mask to the images
original_masked = apply_mask(original_crop, mask)
tri_masked = apply_mask(tri_gray, mask)
spline_masked = apply_mask(spline_gray, mask)

# Convert to float32 for metric calculation
original_masked = original_masked.astype(np.float32)
tri_masked = tri_masked.astype(np.float32)
spline_masked = spline_masked.astype(np.float32)

# Create a valid pixel mask (exclude NaN)
valid_mask = (~np.isnan(original_masked)) & (~np.isnan(tri_masked)) & (~np.isnan(spline_masked))

# Fill invalid areas with zeros
original_valid = np.where(valid_mask, original_masked, 0)
tri_valid = np.where(valid_mask, tri_masked, 0)
spline_valid = np.where(valid_mask, spline_masked, 0)

# Compute SSIM and PSNR metrics
ssim_tri = ssim(original_masked, tri_masked, data_range=255)
ssim_spline = ssim(original_masked, spline_masked, data_range=255)

psnr_tri = psnr(original_masked, tri_masked, data_range=255)
psnr_spline = psnr(original_masked, spline_masked, data_range=255)

# Output the metric results
print(f"Triangular Interpolation - SSIM: {ssim_tri:.4f}, PSNR: {psnr_tri:.2f} dB")
print(f"Cubic Spline Interpolation - SSIM: {ssim_spline:.4f}, PSNR: {psnr_spline:.2f} dB")

# Visualization: side-by-side comparison
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].imshow(original_masked, cmap='gray')
axs[0].set_title("Original (Cropped + Masked)")
axs[1].imshow(tri_masked, cmap='gray')
axs[1].set_title("Triangular Interpolation")
axs[2].imshow(spline_masked, cmap='gray')
axs[2].set_title("Cubic Spline Interpolation")
for ax in axs:
    ax.axis('off')
plt.tight_layout()
plt.show()
