# Vertical Colored Line

* **raw/** – Original clean videos or frames.
* **distortion/** – Frames/videos with synthetic vertical distortion added.
* **denoise/** – Restored outputs after the detection + inpainting pipeline.
* **debug/** – Debug visualizations (e.g., detected line positions).
* **add_distortion.ipynb** – Generates vertical artifacts for testing.
* **denoise.ipynb** – Runs the full denoising + inpainting workflow.
* **evaluate.py** – Computes MSE/PSNR/SSIM to compare raw, distorted, and denoised results.

