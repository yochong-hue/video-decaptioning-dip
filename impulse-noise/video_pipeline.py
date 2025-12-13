# ================================================================
# video_pipeline.py
# Modular pipeline for: mask extraction, noise, denoise, comparison videos
# ================================================================

import os
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm.notebook import tqdm
import sys

# ================================================================
# GLOBAL CONFIG
# ================================================================

SHARED_ROOT = os.path.expanduser("~/teams/imagenie")
MASK_MODEL_PATH = f"{SHARED_ROOT}/Video_Decaptioning/mask_extraction/checkpoint/MaskExtractor.pth"
VIDEO_DECAPTION_PROJ_ROOT = f"{SHARED_ROOT}/Video_Decaptioning"

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

FRAME_SIZE = (128, 128)
DEFAULT_FPS = 25

_mask_net = None  # cache UNet model


# ================================================================
# NAMING HELPERS
# ================================================================

def get_name(video_path):
    return os.path.splitext(os.path.basename(os.path.expanduser(video_path)))[0]


def name_video_original(name): return f"{name}_video_original.mp4"
def name_video_noisy(name):    return f"{name}_video_noisy.mp4"
def name_video_denoised(name, method): 
    return f"{name}_video_denoised_{method}.mp4"

def name_mask_original(name):  return f"{name}_mask_original.mp4"
def name_mask_noisy(name):     return f"{name}_mask_noisy.mp4"
def name_mask_denoised(name, method):
    return f"{name}_mask_denoised_{method}.mp4"

def name_compare_video(name):  return f"{name}_compare_video.mp4"
def name_compare_mask(name):   return f"{name}_compare_mask.mp4"


# ================================================================
# LOAD MASK EXTRACTOR
# ================================================================

def load_mask_extractor():
    global _mask_net
    if _mask_net is not None:
        return _mask_net

    print("[INFO] Loading MaskExtractor UNet...")
    sys.path.append(VIDEO_DECAPTION_PROJ_ROOT)
    from mask_extraction.model.network import UNet

    net = UNet(n_channels=3, n_classes=1)
    net.load_state_dict(torch.load(MASK_MODEL_PATH, map_location="cuda"))
    net = net.cuda()
    net = torch.nn.DataParallel(net, device_ids=[0])
    net.eval()

    _mask_net = net
    return net


# ================================================================
# VIDEO I/O
# ================================================================

def read_video(video_path, resize_to=None):
    video_path = os.path.expanduser(video_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = DEFAULT_FPS

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if resize_to:
            frame = cv2.resize(frame, resize_to)
        frames.append(frame.astype(np.uint8))

    cap.release()
    return frames, fps

# use mp4
def write_video(frames, out_path, fps):
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    for f in frames:
        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))

    writer.release()

# # use lossless avi
# def write_video(frames, out_path, fps):
#     h, w = frames[0].shape[:2]
#     fourcc = cv2.VideoWriter_fourcc(*"FFV1")   # lossless
#     writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

#     for f in frames:
#         writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
#     writer.release()


# ================================================================
# NOISE GENERATOR
# ================================================================

def add_salt_pepper(frame, amount=0.002):
    noisy = frame.copy()
    h, w, _ = frame.shape
    num = int(amount * h * w)

    coords = (np.random.randint(0, h, num), np.random.randint(0, w, num))
    noisy[coords] = [255, 255, 255]

    coords = (np.random.randint(0, h, num), np.random.randint(0, w, num))
    noisy[coords] = [0, 0, 0]

    return noisy

def add_realistic_gaussian_sensor_noise(frame, noise_level=0.04):
    """
    Realistic camera sensor noise:
    - Poisson (shot) noise
    - Gaussian (read) noise
    - Slight RGB correlation
    """
    img = frame.astype(np.float32) / 255.0
    
    # ---- 1. Shot noise (Poisson) ----
    shot = np.random.poisson(img * 255.0) / 255.0
    shot = shot.astype(np.float32)

    # ---- 2. Read noise (Gaussian) ----
    gauss = np.random.normal(0, noise_level, img.shape).astype(np.float32)

    # ---- 3. Correlation (RGB shared shift) ----
    corr = np.random.normal(0, noise_level/2, (img.shape[0], img.shape[1], 1))
    corr = np.repeat(corr, 3, axis=2)

    noisy = img + (shot - img) * 0.5 + gauss + corr
    noisy = np.clip(noisy, 0, 1)
    
    return (noisy * 255).astype(np.uint8)

# def add_flicker(frame, strength=0.4):
#     """
#     Single-frame exposure flicker:
#     - Realistic per-frame brightness variation
#     - Uses multiplicative gain
#     """
#     gain = 1.0 + np.random.uniform(-strength, strength)

#     f = frame.astype(np.float32)
#     f_mod = np.clip(f * gain, 0, 255)
#     return f_mod.astype(np.uint8)

def add_edge_dropout_noise(frame, drop_prob=0.15, drop_strength=0.6):
    """
    Realistic 'edge dropout' noise:
    - selectively weakens random edges
    - looks natural (like low bitrate compression)
    - breaks CNN subtitle detection strongly
    - fully reversible by DIP denoising
    """
    h, w, _ = frame.shape
    f = frame.astype(np.float32) / 255.0

    # --- 1. Detect edges (Sobel magnitude) ---
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx*gx + gy*gy)

    # Normalize edge map
    mag = mag / (mag.max() + 1e-6)

    # --- 2. Random mask over edges ---
    # Only choose pixels that are edges AND random enough
    rand_mask = np.random.rand(h, w)
    dropout_mask = (mag > 0.3) & (rand_mask < drop_prob)

    # --- 3. Apply dropout by weakening the edge pixels ---
    # This reduces stroke intensity but keeps natural look
    weakened = f.copy()
    weakened[dropout_mask] *= drop_strength  # reduce brightness (erosion-like)

    weakened = np.clip(weakened * 255, 0, 255).astype(np.uint8)
    return weakened


def add_moving_moire(frame, t, strength=0.45):
    h, w, _ = frame.shape
    img = frame.astype(np.float32) / 255.0

    # normalized coords
    y = np.linspace(0, 1, h)
    x = np.linspace(0, 1, w)
    xx, yy = np.meshgrid(x, y)

    # ============================================================
    # 1) LARGE GLOBAL CURVE (this creates penguin-like bending)
    # ============================================================
    big_curve = (
        0.40 * np.sin(2*np.pi*(yy*0.25 + 0.02*t)) +   # vertical bend
        0.25 * np.sin(2*np.pi*(xx*0.18 + 0.015*t))    # horizontal bend
    )

    # Warp coordinates with big curve
    u = xx + big_curve
    v = yy + 0.8 * big_curve

    # ============================================================
    # 2) IRREGULAR SPACING (wavelength drift)
    # ============================================================
    # frequency slowly varies across frame
    fx = 12 + 4*np.sin(2*np.pi*(v*0.6 + 0.01*t))
    fy = 14 + 5*np.sin(2*np.pi*(u*0.5 + 0.015*t))

    base_wave = fx*u + fy*v

    # ============================================================
    # 3) RGB SUBPIXEL OFFSETS (rainbow)
    # ============================================================
    r_phase = 0.10*np.sin(2*np.pi*(0.005*t + u*0.8))
    g_phase = 0.10*np.sin(2*np.pi*(0.006*t + v*0.7))
    b_phase = 0.10*np.sin(2*np.pi*(0.004*t + u*0.6 + v*0.2))

    R = np.sin(2*np.pi*(base_wave*1.00 + r_phase))
    G = np.sin(2*np.pi*(base_wave*1.03 + g_phase))
    B = np.sin(2*np.pi*(base_wave*0.97 + b_phase))

    moire = np.stack([R, G, B], axis=2)

    # normalize 0–1
    moire = (moire - moire.min()) / (moire.max() - moire.min() + 1e-6)

    # boost
    moire = 0.5 + (moire - 0.5)*1.7
    moire = np.clip(moire, 0, 1)

    # apply to image
    out = img * (1 - strength*(moire - 0.5))
    out = np.clip(out * 255, 0, 255).astype(np.uint8)
    return out

def add_correct_lcd_moire(frame, t, strength=0.35):
    """
    Correct LCD-camera moire:
    - one dominant diagonal orientation
    - gentle curvature only (NO circular waves)
    - slight frequency drift (irregular spacing)
    - subtle chromatic fringes along stripe edges 
    - realistic movement
    """

    img = frame.astype(np.float32) / 255.0
    h, w, _ = img.shape

    # normalized coords
    y = np.linspace(0, 1, h)
    x = np.linspace(0, 1, w)
    xx, yy = np.meshgrid(x, y)

    # --------------------------------------------------
    # 1) Dominant diagonal wave (primary moire direction)
    # --------------------------------------------------
    # diagonal grid: (xx*cosθ + yy*sinθ)
    theta = -0.65   # ~ -37°, realistic LCD moire slope
    grid = xx*np.cos(theta) + yy*np.sin(theta)

    # Base moire
    f = 22 + 4*np.sin(2*np.pi*(0.01*t))  # slowly changing freq
    base = f * grid

    # --------------------------------------------------
    # 2) GLOBAL CURVATURE (small bend, NOT circular)
    # --------------------------------------------------
    # controls curvature amplitude (very important!)
    curve_amp = 0.13  

    curvature = curve_amp * np.sin(2*np.pi*(yy*0.25 + 0.02*t))

    # CURVED but NOT radial
    phase = base + curvature

    # --------------------------------------------------
    # 3) RGB chromatic fringing (slight, realistic)
    # --------------------------------------------------
    R = np.sin(2*np.pi*(phase + 0.02*np.sin(xx*5)))
    G = np.sin(2*np.pi*(phase + 0.03*np.sin(xx*6)))
    B = np.sin(2*np.pi*(phase + 0.05*np.sin(xx*7)))

    moire = np.stack([R, G, B], axis=2)

    # normalize
    moire = (moire - moire.min()) / (moire.max() - moire.min() + 1e-6)

    # subtle moire only → no huge rainbows or circles
    moire = 0.5 + (moire - 0.5)*1.2
    moire = np.clip(moire, 0, 1)

    # blend
    out = img * (1 - strength*(moire - 0.5))
    out = np.clip(out*255, 0, 255).astype(np.uint8)

    return out

def apply_noise(frames):
    return [add_salt_pepper(f) for f in frames]
    # return [add_realistic_gaussian_sensor_noise(f) for f in frames]
    # return [add_flicker(f) for f in frames]  # doesn't work, CNN is per frame
    # return [add_edge_dropout_noise(f) for f in frames]
    # return [add_realistic_moire(f) for f in frames]

    # out = []
    # for i, frame in enumerate(frames):
    #     noisy_frame = add_moving_moire(frame, t=i / 30)  # looks good but hard to recover

    #     # noisy_frame = add_correct_lcd_moire(frame, t=i / 30)
    #     out.append(noisy_frame)
    # return out
    
    


# ================================================================
# DENOISING
# ================================================================

# def denoise_frame(frame, method="median"):
#     if method == "median":
#         med = cv2.medianBlur(frame, 3)
#         kernel = np.ones((2,2), np.uint8)
#         opened = cv2.morphologyEx(med, cv2.MORPH_OPEN, kernel)
#         return opened
#     else:
#         raise ValueError(f"Unknown method: {method}")

def denoise_adaptive_median(frame):
    # Step 1: detect impulse candidates
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    blur3 = cv2.medianBlur(gray, 3)
    noise_mask = (cv2.absdiff(gray, blur3) > 30)

    cleaned = frame.copy()

    # Step 2: adaptive masked median
    for k in [3, 5, 7]:
        med = cv2.medianBlur(frame, k)
        noise_mask_3ch = np.repeat(noise_mask[:, :, None], 3, axis=2)
        still_bad = noise_mask_3ch & (np.abs(med.astype(int) - frame.astype(int)) > 15)
        cleaned[still_bad] = med[still_bad]
        if not np.any(still_bad):
            break

    # Step 3: structural reconnection
    kernel = np.ones((2,2), np.uint8)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

    return cleaned

def denoise_median_morph_0(frame):
    med = cv2.medianBlur(frame, 3)
    kernel = np.ones((2,2), np.uint8)
    opened = cv2.morphologyEx(med, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    return closed

def denoise_median_thresh_morph(frame):
    """
    Median → Adaptive Threshold → Morphological Opening & Closing
    Designed to:
    - remove impulse noise
    - separate subtitle strokes locally under weak contrast
    - reconnect fragmented strokes
    """
    # 1. Remove isolated impulses
    med = cv2.medianBlur(frame, 3)

    # 2. Convert to grayscale for adaptive thresholding
    gray = cv2.cvtColor(med, cv2.COLOR_RGB2GRAY)

    # 3. Adaptive Gaussian thresholding
    thresh = cv2.adaptiveThreshold(
        gray,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=15,         # Local region (tunable per subtitle stroke size)
        C=2                   # Fine adjustment
    )

    # 4. Morphological refinement using same kernel
    kernel = np.ones((3, 3), np.uint8)  # same structuring element
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

    return cv2.cvtColor(closed, cv2.COLOR_GRAY2RGB)

def denoise_median_morph(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    diff = cv2.absdiff(gray, cv2.medianBlur(gray, 3))
    impulse_mask = diff > 30  # tweak if needed

    med_frame = cv2.medianBlur(frame, 3)
    cleaned = frame.copy()
    cleaned[impulse_mask] = med_frame[impulse_mask]

    kernel = np.ones((2,2), np.uint8)
    closed = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

    final = cv2.bilateralFilter(closed, d=5, sigmaColor=20, sigmaSpace=5)
    return final

# def denoise_frame_enhanced(frame):
#     # 1. Remove Gaussian + Poisson noise (strong)
#     f1 = cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 21)
    
#     # 2. Remove impulse noise (very light)
#     f2 = cv2.medianBlur(f1, 3)
    
#     # 3. Preserve edges + final polish
#     f3 = cv2.bilateralFilter(f2, d=5, sigmaColor=25, sigmaSpace=5)
    
#     return f3

# def denoise_frame_flicker(frame, target_mean=128):
#     """
#     Single-frame exposure normalization:
#     - Computes per-frame gain
#     - Normalizes brightness to target mean
#     """
#     f = frame.astype(np.float32)
#     current_mean = f.mean()

#     gain = target_mean / (current_mean + 1e-6)
#     corrected = np.clip(f * gain, 0, 255)
#     return corrected.astype(np.uint8)

# def denoise_edge_dropout(frame):
#     """
#     DIP-style restoration for edge-dropout noise:
#     - unsharp mask to restore edges
#     - morphological gap filling for thin strokes
#     - bilateral filter smoothing to remove artifacts
#     """
#     f = frame.astype(np.float32) / 255.0

#     # --- 1. Unsharp mask (light sharpening to restore edges) ---
#     blurred = cv2.GaussianBlur(f, (5, 5), 1.2)
#     sharp = cv2.addWeighted(f, 1.7, blurred, -0.7, 0)

#     sharp = np.clip(sharp, 0, 1)

#     # --- 2. Morphological closing on grayscale (fills gaps in subtitle strokes) ---
#     gray = (sharp.mean(axis=2) * 255).astype(np.uint8)
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
#     closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

#     # Combine closed grayscale back with color information
#     closed_rgb = cv2.cvtColor(closed, cv2.COLOR_GRAY2RGB) / 255.0
#     restored = (0.6 * sharp + 0.4 * closed_rgb)

#     # --- 3. Bilateral filter (edge-preserving polish) ---
#     restored = np.clip(restored * 255, 0, 255).astype(np.uint8)
#     restored = cv2.bilateralFilter(restored, d=5, sigmaColor=35, sigmaSpace=5)

#     return restored


# def denoise_moire_fft(frame):
#     """
#     Robust moire removal using a directional soft-notch filter.
#     No Hessian, no Sobel, no OpenCV issues.
#     Works for curved diagonal moire in 128x128 LCD videos.
#     """
#     img = frame.astype(np.float32) / 255.0
#     h, w, c = img.shape

#     out = np.zeros_like(img)

#     # Create coordinate grid for frequency domain
#     u = np.linspace(-1, 1, w)
#     v = np.linspace(-1, 1, h)
#     uu, vv = np.meshgrid(u, v)

#     # Direction of the moire (diagonal)
#     theta = np.deg2rad(-45)  # adjust if needed
#     dir_mask = uu * np.cos(theta) + vv * np.sin(theta)

#     # Curved soft notch for removing the band
#     # A Gaussian band-stop filter
#     notch = 1 - np.exp(- (dir_mask**2) * 15.0)

#     # Slight curvature compensation
#     curvature = np.exp(-(vv**2) * 4.0)
#     notch = notch * curvature + (1 - curvature)

#     # Ensure notch ranges between 0.2 and 1.0
#     notch = 0.2 + 0.8 * notch
#     notch = notch.astype(np.float32)

#     for ch in range(3):
#         I = img[..., ch]

#         # FFT
#         F = np.fft.fftshift(np.fft.fft2(I))

#         # Apply the notch
#         F_filtered = F * notch

#         # Inverse FFT
#         clean = np.real(np.fft.ifft2(np.fft.ifftshift(F_filtered)))
#         clean = np.clip(clean, 0, 1)

#         out[..., ch] = clean

#     return (out * 255).astype(np.uint8)


# def denoise_video_frames(frames, method="median"):
#     # return [denoise_frame(f, method) for f in frames]
#     # return [denoise_frame_enhanced(f) for f in frames]
#     # return [denoise_frame_flicker(f) for f in frames]  # doesn't work as CNN is per frame
#     # return [denoise_edge_dropout(f) for f in frames]
#     return [denoise_moire_fft(f) for f in frames]

def denoise_video_frames(frames, method="median-morph"):
    if method == "adaptive-median":
        return [denoise_adaptive_median(f) for f in frames]
    elif method == "median-morph":
        return [denoise_median_morph(f) for f in frames]
    elif method == "median-thresh-morph":
        return [denoise_median_thresh_morph(f) for f in frames]



# ================================================================
# MASK EXTRACTION
# ================================================================

def extract_mask_frame_tensor(frames):
    net = load_mask_extractor()

    frames_np = np.transpose(np.stack(frames), (0,3,1,2))
    frames_np = (frames_np / 255.0 - 0.5) / 0.5

    t = torch.from_numpy(frames_np).float().cuda().unsqueeze(0)

    with torch.no_grad():
        masks = net(t).squeeze(0)
        masks = (masks > 0.5).float()

    to_pil = transforms.ToPILImage()
    out = []
    for i in range(masks.shape[0]):
        p = to_pil(masks[i].cpu()).convert("RGB")
        out.append(np.array(p))
    return out


def extract_mask_video(video_path, out_path):
    frames, fps = read_video(video_path)
    mask_frames = extract_mask_frame_tensor(frames)
    write_video(mask_frames, out_path, fps)
    return out_path


# ================================================================
# SIDE BY SIDE (3-COLUMN)
# ================================================================

def side_by_side_3(v1_frames, v2_frames, v3_frames):
    out = []
    for a, b, c in zip(v1_frames, v2_frames, v3_frames):
        out.append(np.hstack([a, b, c]))
    return out


def compare_video_row(name, orig_path, noisy_path, den_path):
    f1, fps = read_video(orig_path)
    f2, _   = read_video(noisy_path)
    f3, _   = read_video(den_path)

    combo = side_by_side_3(f1, f2, f3)
    out_path = name_compare_video(name)
    write_video(combo, out_path, fps)
    return out_path


def compare_mask_row(name, orig_mask_path, noisy_mask_path, den_mask_path):
    f1, fps = read_video(orig_mask_path)
    f2, _   = read_video(noisy_mask_path)
    f3, _   = read_video(den_mask_path)

    combo = side_by_side_3(f1, f2, f3)
    out_path = name_compare_mask(name)
    write_video(combo, out_path, fps)
    return out_path


# ================================================================
# FULL PIPELINE
# ================================================================

def full_process(video_path, denoise_method="median-morph"):
    name = get_name(video_path)

    # 1. Original video → mask
    orig_video = name_video_original(name)
    os.system(f"cp {os.path.expanduser(video_path)} {orig_video}")
    orig_mask = name_mask_original(name)
    extract_mask_video(video_path, orig_mask)

    # 2. Noisy video → mask
    frames, fps = read_video(video_path)
    noisy_frames = apply_noise(frames)

    # -------------------------------------------
    # DEBUG: save noisy frames to PNGs
    # -------------------------------------------
    debug_dir = f"_debug_noisy_frames/{name}"
    os.makedirs(debug_dir, exist_ok=True)
    
    for i, f in enumerate(noisy_frames):
        cv2.imwrite(
            os.path.join(debug_dir, f"{i:04d}.png"),
            cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
        )
    print(f"[DEBUG] Saved noisy frames → {debug_dir}")
    
    # -------------------------------------------
    noisy_video = name_video_noisy(name)
    write_video(noisy_frames, noisy_video, fps)

    noisy_mask = name_mask_noisy(name)
    extract_mask_video(noisy_video, noisy_mask)

    # 3. Denoised video → mask
    den_frames = denoise_video_frames(noisy_frames, method=denoise_method)
    den_video = name_video_denoised(name, denoise_method)
    write_video(den_frames, den_video, fps)

    den_mask = name_mask_denoised(name, denoise_method)
    extract_mask_video(den_video, den_mask)

    # 4. Comparison rows
    comp_video = compare_video_row(name, orig_video, noisy_video, den_video)
    comp_mask  = compare_mask_row(name, orig_mask, noisy_mask, den_mask)

    return {
        "original_video": orig_video,
        "noised_video": noisy_video,
        "denoised_video": den_video,
        "original_mask": orig_mask,
        "noised_mask": noisy_mask,
        "denoised_mask": den_mask,
        "compare_video": comp_video,
        "compare_mask": comp_mask,
    }