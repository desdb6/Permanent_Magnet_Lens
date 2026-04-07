"""
claude_radon.py
==================
Scientifically robust measurement of square sizes in noisy grayscale images
stored as space/tab/comma-separated text files.
 
Methods
-------
1. Radon Transform  – robust to rotation and moderate noise; works on a single square.
2. 2D Autocorrelation – works best when multiple squares tile the image (noise averages out).
 
Usage
-----
    python measure_squares.py image.txt [--method radon|autocorr|both] [--plot]
"""
 
import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
 
 
# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
 
def load_image(path: str) -> np.ndarray:
    """Load a grayscale image from a plain-text file.
 
    Accepts space, tab, or comma-separated values, with or without a header.
    Returns a 2-D float64 array normalised to [0, 1].
    """
    path = Path(path)
    if not path.exists():
        sys.exit(f"File not found: {path}")
 
    # Try numpy's generic loader first (handles whitespace separators)
    for delim in [None, ",", ";"]:
        try:
            img = np.loadtxt(path, delimiter=delim)
            if img.ndim == 2:
                break
        except Exception:
            continue
    else:
        sys.exit("Could not parse the file. Make sure it is a 2-D numeric text array.")
 
    img = img.astype(np.float64)
    lo, hi = img.min(), img.max()
    if hi > lo:
        img = (img - lo) / (hi - lo)
    print(f"Loaded image: {img.shape[0]} rows × {img.shape[1]} cols  "
          f"(value range {lo:.3g} – {hi:.3g})")
    return img
 
 
# ---------------------------------------------------------------------------
# Pre-processing
# ---------------------------------------------------------------------------
 
def preprocess(img: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    """Mild Gaussian blur to suppress pixel-level noise before transforms."""
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(img, sigma=sigma)
 
 
# ---------------------------------------------------------------------------
# Method 1 – Radon Transform
# ---------------------------------------------------------------------------
 
def radon_transform(img: np.ndarray, n_angles: int = 360) -> tuple[np.ndarray, np.ndarray]:
    """Compute the Radon transform (sinogram) of *img*.
 
    Uses a pure-NumPy implementation so scikit-image is not required,
    but will use skimage if available (faster & more accurate).
    """
    try:
        from skimage.transform import radon
        theta = np.linspace(0, 180, n_angles, endpoint=False)
        sinogram = radon(img, theta=theta, circle=False)
        return sinogram, theta
    except ImportError:
        pass  # fall back to NumPy implementation below
 
    # --- Pure NumPy fallback ---
    h, w = img.shape
    diag = int(np.ceil(np.hypot(h, w)))
    pad_h = (diag - h) // 2
    pad_w = (diag - w) // 2
    padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode="constant")
 
    theta = np.linspace(0, 180, n_angles, endpoint=False)
    sinogram = np.zeros((padded.shape[1], n_angles))
    cy, cx = np.array(padded.shape) / 2
 
    for i, angle in enumerate(theta):
        rad = np.deg2rad(angle)
        cos_a, sin_a = np.cos(rad), np.sin(rad)
        ys, xs = np.indices(padded.shape)
        proj_coords = (xs - cx) * cos_a + (ys - cy) * sin_a
        bins = np.round(proj_coords + padded.shape[1] / 2).astype(int)
        proj = np.bincount(bins.ravel(), weights=padded.ravel(),
                           minlength=padded.shape[1])
        sinogram[:, i] = proj
 
    return sinogram, theta
 
 
def measure_by_radon(img: np.ndarray, plot: bool = False) -> dict:
    """
    Measure square size from the Radon transform.
 
    Strategy
    --------
    * Compute the 1-D power spectrum of each angular projection.
    * A square edge pair produces two parallel lines → a dominant frequency
      in the projection whose inverse gives the square size.
    * We pick the angle that maximises the peak-to-noise ratio in the
      projection spectrum, then read off the period.
 
    Returns
    -------
    dict with keys: angle_deg, size_px, size_uncertainty_px
    """
    print("\n─── Radon Transform Method ───")
    img_proc = preprocess(img)
    sinogram, theta = radon_transform(img_proc)
 
    n_proj, n_angles = sinogram.shape
 
    best_snr = -np.inf
    best_angle_idx = 0
    best_period = None
 
    # Evaluate every angular projection
    snr_curve = np.zeros(n_angles)
    period_curve = np.zeros(n_angles)
 
    for i in range(n_angles):
        proj = sinogram[:, i]
        proj -= proj.mean()
 
        # Power spectrum of this projection
        ft = np.abs(np.fft.rfft(proj)) ** 2
        freqs = np.fft.rfftfreq(len(proj))
 
        # Ignore DC and very low frequencies (global gradients)
        # and very high frequencies (< 3 px period = pure noise)
        valid = (freqs > 1 / (len(proj) * 0.9)) & (freqs < 1 / 3)
        if not valid.any():
            continue
 
        ft_valid = ft.copy()
        ft_valid[~valid] = 0
 
        peak_idx = np.argmax(ft_valid)
        if freqs[peak_idx] == 0:
            continue
 
        period = 1.0 / freqs[peak_idx]
        noise_floor = np.median(ft[valid])
        snr = ft[peak_idx] / (noise_floor + 1e-12)
 
        snr_curve[i] = snr
        period_curve[i] = period
 
        if snr > best_snr:
            best_snr = snr
            best_angle_idx = i
            best_period = period
 
    best_angle = theta[best_angle_idx]
 
    # Uncertainty: spread of period estimates among high-SNR angles
    threshold = 0.5 * best_snr
    good = snr_curve > threshold
    periods_good = period_curve[good]
    uncertainty = periods_good.std() if good.sum() > 1 else np.nan
 
    print(f"  Best projection angle : {best_angle:.1f}°  "
          f"(square edge is ≈ {best_angle:.1f}° from horizontal)")
    print(f"  Estimated square size : {best_period:.2f} px")
    print(f"  Uncertainty (1-σ)     : {uncertainty:.2f} px")
    print(f"  Peak SNR              : {best_snr:.1f}")
 
    if plot:
        _plot_radon(img, sinogram, theta, best_angle_idx, snr_curve,
                    period_curve, best_period, best_angle)
 
    return {
        "method": "Radon",
        "angle_deg": best_angle,
        "size_px": best_period,
        "size_uncertainty_px": uncertainty,
        "snr": best_snr,
    }
 
 
def _plot_radon(img, sinogram, theta, best_idx, snr_curve, period_curve,
                best_period, best_angle):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Radon Transform Analysis", fontsize=14)
 
    axes[0, 0].imshow(img, cmap="gray", origin="upper")
    axes[0, 0].set_title("Input image")
    axes[0, 0].axis("off")
 
    axes[0, 1].imshow(sinogram, cmap="inferno", aspect="auto",
                      extent=[theta[0], theta[-1], sinogram.shape[0], 0])
    axes[0, 1].axvline(theta[best_idx], color="cyan", lw=1.5, label=f"{best_angle:.1f}°")
    axes[0, 1].set_title("Sinogram (Radon transform)")
    axes[0, 1].set_xlabel("Angle (°)")
    axes[0, 1].set_ylabel("Projection coordinate (px)")
    axes[0, 1].legend(fontsize=8)
 
    axes[1, 0].plot(theta, snr_curve, lw=1)
    axes[1, 0].axvline(theta[best_idx], color="red", lw=1.5,
                       label=f"Best: {best_angle:.1f}°")
    axes[1, 0].set_title("SNR of dominant frequency vs. angle")
    axes[1, 0].set_xlabel("Projection angle (°)")
    axes[1, 0].set_ylabel("SNR")
    axes[1, 0].legend(fontsize=8)
 
    proj = sinogram[:, best_idx]
    ft = np.abs(np.fft.rfft(proj - proj.mean())) ** 2
    freqs = np.fft.rfftfreq(len(proj))
    valid = freqs > 0
    axes[1, 1].plot(1 / freqs[valid][1:], ft[valid][1:], lw=1)
    axes[1, 1].axvline(best_period, color="red", lw=1.5,
                       label=f"Size ≈ {best_period:.1f} px")
    axes[1, 1].set_title(f"Power spectrum – best projection ({best_angle:.1f}°)")
    axes[1, 1].set_xlabel("Period (px)")
    axes[1, 1].set_ylabel("Power")
    axes[1, 1].set_xlim(0, len(proj) // 2)
    axes[1, 1].legend(fontsize=8)
 
    plt.tight_layout()
    plt.savefig("radon_result.png", dpi=150)
    print("  → Saved radon_result.png")
    plt.show()
 
 
# ---------------------------------------------------------------------------
# Method 2 – 2-D Autocorrelation
# ---------------------------------------------------------------------------
 
def measure_by_autocorrelation(img: np.ndarray, plot: bool = False) -> dict:
    """
    Measure square size via 2-D autocorrelation.
 
    Strategy
    --------
    * Compute the normalised 2-D autocorrelation via FFT (Wiener–Khinchin theorem).
    * The central peak is always at (0, 0); secondary peaks at ±(dx, dy) indicate
      the repeat vector of any periodic structure.
    * We find the first significant off-centre peak, compute its distance from
      the origin, and report that as the square period/size.
 
    Best used when the image contains multiple squares arranged in a grid.
    """
    print("\n─── 2-D Autocorrelation Method ───")
    img_proc = preprocess(img)
 
    # Zero-mean, window to suppress edge ringing
    img_zm = img_proc - img_proc.mean()
    window = np.outer(np.hanning(img_zm.shape[0]),
                      np.hanning(img_zm.shape[1]))
    img_zm *= window
 
    # Autocorrelation via FFT (Wiener–Khinchin)
    F = np.fft.fft2(img_zm)
    acf = np.fft.ifft2(F * np.conj(F)).real
    acf = np.fft.fftshift(acf)
    acf /= acf.max()   # normalise so central peak = 1
 
    cy, cx = np.array(acf.shape) // 2
 
    # Mask the central peak (within 3 px) before searching for secondary peaks
    mask_r = 3
    y_grid, x_grid = np.indices(acf.shape)
    dist_from_centre = np.hypot(y_grid - cy, x_grid - cx)
    acf_masked = acf.copy()
    acf_masked[dist_from_centre < mask_r] = 0
 
    # Find the global maximum of the masked ACF
    peak_flat = np.argmax(acf_masked)
    py, px = np.unravel_index(peak_flat, acf.shape)
    dy, dx = py - cy, px - cx
    distance = np.hypot(dy, dx)
    angle = np.degrees(np.arctan2(dy, dx))
    peak_value = acf[py, px]
 
    # Uncertainty: half-width at half-maximum of the secondary peak
    region_r = max(5, int(distance * 0.2))
    y0 = max(0, py - region_r)
    y1 = min(acf.shape[0], py + region_r + 1)
    x0 = max(0, px - region_r)
    x1 = min(acf.shape[1], px + region_r + 1)
    region = acf[y0:y1, x0:x1]
    hwhm_mask = region > peak_value * 0.5
    hwhm_area = hwhm_mask.sum()
    uncertainty = np.sqrt(hwhm_area / np.pi)  # equivalent radius
 
    print(f"  Repeat vector         : ({dx:+.1f}, {dy:+.1f}) px  "
          f"→ angle {angle:.1f}°")
    print(f"  Estimated square size : {distance:.2f} px")
    print(f"  Uncertainty (HWHM)    : ±{uncertainty:.2f} px")
    print(f"  Secondary peak height : {peak_value:.3f}  "
          f"(1.0 = perfect periodicity)")
 
    if plot:
        _plot_autocorr(img, acf, cy, cx, py, px, distance, peak_value)
 
    return {
        "method": "Autocorrelation",
        "repeat_vector_px": (float(dx), float(dy)),
        "angle_deg": float(angle),
        "size_px": float(distance),
        "size_uncertainty_px": float(uncertainty),
        "peak_strength": float(peak_value),
    }
 
 
def _plot_autocorr(img, acf, cy, cx, py, px, distance, peak_value):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("2-D Autocorrelation Analysis", fontsize=14)
 
    axes[0].imshow(img, cmap="gray", origin="upper")
    axes[0].set_title("Input image")
    axes[0].axis("off")
 
    half = min(cy, cx, 80)  # show central ±80 px of ACF
    acf_crop = acf[cy - half:cy + half, cx - half:cx + half]
    im = axes[1].imshow(acf_crop, cmap="RdBu_r", origin="upper",
                        vmin=-0.5, vmax=1, extent=[-half, half, half, -half])
    axes[1].scatter([px - cx], [py - cy], s=80, c="yellow",
                    marker="x", linewidths=2, label=f"Peak ({distance:.1f} px)")
    axes[1].set_title("2-D Autocorrelation (centre crop)")
    axes[1].set_xlabel("Δx (px)")
    axes[1].set_ylabel("Δy (px)")
    axes[1].legend(fontsize=8)
    plt.colorbar(im, ax=axes[1], fraction=0.046)
 
    # Radial profile
    max_r = min(cy, cx)
    r_vals = np.arange(1, max_r)
    radial = np.array([
        acf[cy - r:cy + r + 1, cx - r:cx + r + 1].mean()
        for r in r_vals
    ])
    axes[2].plot(r_vals, radial, lw=1.2)
    axes[2].axvline(distance, color="red", lw=1.5,
                    label=f"Size ≈ {distance:.1f} px")
    axes[2].set_title("Radial profile of ACF")
    axes[2].set_xlabel("Radius (px)")
    axes[2].set_ylabel("Mean ACF value")
    axes[2].legend(fontsize=8)
 
    plt.tight_layout()
    plt.savefig("autocorr_result.png", dpi=150)
    print("  → Saved autocorr_result.png")
    plt.show()
 
 
# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
 
def print_summary(results: list[dict]):
    print("\n" + "═" * 50)
    print("  SUMMARY")
    print("═" * 50)
    sizes = []
    for r in results:
        u = r.get("size_uncertainty_px", float("nan"))
        print(f"  {r['method']:20s}  {r['size_px']:.2f} ± {u:.2f} px")
        sizes.append(r["size_px"])
    if len(sizes) > 1:
        mean_size = np.mean(sizes)
        print(f"  {'Combined estimate':20s}  {mean_size:.2f} px")
    print("═" * 50)
 
 
# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
 
def main():
    parser = argparse.ArgumentParser(
        description="Measure square size in noisy greyscale images (text format).")
    parser.add_argument("image", help="Path to the text-format image file")
    parser.add_argument("--method", choices=["radon", "autocorr", "both"],
                        default="both",
                        help="Measurement method (default: both)")
    parser.add_argument("--plot", action="store_true",
                        help="Show and save diagnostic plots")
    parser.add_argument("--sigma", type=float, default=1.5,
                        help="Pre-processing Gaussian blur sigma (default 1.5)")
    args = parser.parse_args()
 
    img = load_image(args.image)
 
    results = []
    if args.method in ("radon", "both"):
        results.append(measure_by_radon(img, plot=args.plot))
    if args.method in ("autocorr", "both"):
        results.append(measure_by_autocorrelation(img, plot=args.plot))
 
    print_summary(results)
 
 
if __name__ == "__main__":
    main()