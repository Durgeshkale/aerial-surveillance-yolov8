"""
Image Enhancement Module for Aerial Surveillance
Applies CLAHE, denoising, sharpening and histogram equalization
to improve aerial image quality before inference.
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import os


def clahe_enhance(img_bgr: np.ndarray, clip_limit=2.0, tile_size=(8, 8)) -> np.ndarray:
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge([l_eq, a, b])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)


def denoise(img_bgr: np.ndarray, h=10, template_window=7, search_window=21) -> np.ndarray:
    """Apply Non-local Means Denoising."""
    return cv2.fastNlMeansDenoisingColored(img_bgr, None, h, h, template_window, search_window)


def sharpen(img_bgr: np.ndarray, strength=1.5) -> np.ndarray:
    """Unsharp masking for edge enhancement."""
    gaussian = cv2.GaussianBlur(img_bgr, (9, 9), 10.0)
    sharpened = cv2.addWeighted(img_bgr, 1.0 + strength, gaussian, -strength, 0)
    return sharpened


def gamma_correction(img_bgr: np.ndarray, gamma=1.2) -> np.ndarray:
    """Apply gamma correction for brightness adjustment."""
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(img_bgr, table)


def histogram_equalization(img_bgr: np.ndarray) -> np.ndarray:
    """Histogram equalization on each channel."""
    channels = cv2.split(img_bgr)
    eq_channels = [cv2.equalizeHist(c) for c in channels]
    return cv2.merge(eq_channels)


def enhance_aerial_image(
    img_input,
    apply_clahe: bool = True,
    apply_denoise: bool = True,
    apply_sharpen: bool = True,
    apply_gamma: bool = True,
    gamma_value: float = 1.1,
    sharpen_strength: float = 1.2,
) -> np.ndarray:
    """
    Full enhancement pipeline for aerial imagery.
    
    Args:
        img_input: Path (str) or numpy BGR array
        apply_clahe: Enable CLAHE contrast enhancement
        apply_denoise: Enable denoising
        apply_sharpen: Enable sharpening
        apply_gamma: Enable gamma correction
        gamma_value: Gamma value (>1 brightens, <1 darkens)
        sharpen_strength: Unsharp mask strength

    Returns:
        Enhanced BGR numpy array
    """
    if isinstance(img_input, str):
        img = cv2.imread(img_input)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_input}")
    else:
        img = img_input.copy()

    if apply_denoise:
        img = denoise(img)

    if apply_clahe:
        img = clahe_enhance(img)

    if apply_gamma:
        img = gamma_correction(img, gamma=gamma_value)

    if apply_sharpen:
        img = sharpen(img, strength=sharpen_strength)

    return img


def enhance_and_save(input_path: str, output_path: str, **kwargs) -> None:
    enhanced = enhance_aerial_image(input_path, **kwargs)
    cv2.imwrite(output_path, enhanced)
    print(f"Enhanced image saved to {output_path}")


def compute_image_stats(img_bgr: np.ndarray) -> dict:
    """Compute basic quality metrics for an image."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    mean_brightness = np.mean(gray)
    contrast = gray.std()
    return {
        "sharpness_score": round(laplacian_var, 2),
        "mean_brightness": round(float(mean_brightness), 2),
        "contrast": round(float(contrast), 2),
        "resolution": f"{img_bgr.shape[1]}x{img_bgr.shape[0]}"
    }


if __name__ == "__main__":
    # Demo: enhance a sample image
    import sys
    if len(sys.argv) >= 3:
        inp, out = sys.argv[1], sys.argv[2]
        enhance_and_save(inp, out)
        print("Stats:", compute_image_stats(cv2.imread(out)))
    else:
        print("Usage: python enhance.py <input_image> <output_image>")
        
