import numpy as np
import matplotlib.pyplot as plt
from evio.source.dat_file import DatFileSource
from main import get_window
from rpm_estimator import RpmEstimator

BG = "#031402"
FG = "#71fca5"

def process_file(dat_path: str, window_ms: float = 10.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Support up to 6500 RPM (108.3 Hz), minimum 5 Hz (300 RPM)
    src = DatFileSource(dat_path, width=1280, height=720, window_length_us=window_ms * 1000)
    rpm_est = RpmEstimator(min_hz=5.0, max_hz=110.0, history_s=6.0)
    times = []
    rpms = []
    
    for batch_range in src.ranges():
        window = get_window(
            src.event_words,
            src.order,
            batch_range.start,
            batch_range.stop,
        )
        t_end_s = batch_range.end_ts_us / 1e6
        win_s = src.window_length_us / 1e6 if hasattr(src, "window_length_us") else (window_ms / 1000.0)
        est = rpm_est.update(window, t_end_s, win_s)
        times.append(t_end_s)
        rpms.append(est.rpm if est.rpm is not None else np.nan)
    
    times = np.array(times)
    rpms = np.array(rpms)
    
    # Multi-stage filtering for better predictions
    valid_mask = ~np.isnan(rpms)
    if np.sum(valid_mask) > 5:
        from scipy import signal
        
        # Stage 1: Outlier removal using IQR method
        valid_rpms = rpms[valid_mask]
        q1, q3 = np.percentile(valid_rpms, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 2.0 * iqr
        upper_bound = q3 + 2.0 * iqr
        
        # Replace outliers with NaN
        outlier_mask = (rpms < lower_bound) | (rpms > upper_bound)
        rpms[outlier_mask] = np.nan
        valid_mask = ~np.isnan(rpms)
        
        # Stage 2: Temporal consistency - RPM shouldn't change too rapidly
        if np.sum(valid_mask) > 3:
            valid_indices = np.where(valid_mask)[0]
            valid_rpms = rpms[valid_mask]
            valid_times = times[valid_mask]
            
            # Compute rate of change
            if len(valid_rpms) > 1:
                time_diffs = np.diff(valid_times)
                rpm_diffs = np.diff(valid_rpms)
                rpm_rates = rpm_diffs / (time_diffs + 1e-10)
                
                # Mark points with excessive rate of change (>2000 RPM/s is unrealistic even for high RPM)
                excessive_change = np.abs(rpm_rates) > 2000
                if np.any(excessive_change):
                    # Mark the second point of each excessive change pair
                    for i in range(len(excessive_change)):
                        if excessive_change[i]:
                            idx = valid_indices[i + 1]
                            rpms[idx] = np.nan
                    valid_mask = ~np.isnan(rpms)
            
            # Stage 3: Median filter for robustness (less sensitive to outliers than Savitzky-Golay)
            if np.sum(valid_mask) > 5:
                valid_indices = np.where(valid_mask)[0]
                valid_rpms = rpms[valid_mask]
                
                # Use median filter with adaptive window
                window_size = min(7, len(valid_indices) // 3)
                if window_size >= 3 and window_size % 2 == 1:
                    filtered = signal.medfilt(valid_rpms, kernel_size=window_size)
                    rpms[valid_mask] = filtered
                    
                    # Stage 4: Light smoothing with Savitzky-Golay
                    if len(valid_rpms) > 5:
                        sg_window = min(5, len(valid_indices) // 4)
                        if sg_window >= 3 and sg_window % 2 == 1:
                            smoothed = signal.savgol_filter(rpms[valid_mask], sg_window, 2)
                            rpms[valid_mask] = smoothed
    
    # Compute rate of change (RPM/s) using finite differences
    # Handle NaN by forward-filling for gradient calculation
    rpms_filled = rpms.copy()
    if np.sum(valid_mask) > 0:
        # Forward fill NaN values for gradient calculation
        last_valid = None
        for i in range(len(rpms)):
            if valid_mask[i]:
                last_valid = rpms[i]
            elif last_valid is not None:
                rpms_filled[i] = last_valid
    
    # Compute gradient on all points, but set to NaN where original was NaN
    rpm_rate = np.gradient(rpms_filled, times)
    rpm_rate[~valid_mask] = np.nan
    
    # Interpolate NaN values for continuous plotting
    rpms_plot = rpms.copy()
    rpm_rate_plot = rpm_rate.copy()
    if np.sum(valid_mask) > 1:
        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) > 1:
            # Linear interpolation for RPM
            rpms_plot = np.interp(times, times[valid_mask], rpms[valid_mask])
            # Linear interpolation for rate
            valid_rate_mask = ~np.isnan(rpm_rate)
            if np.sum(valid_rate_mask) > 1:
                rpm_rate_plot = np.interp(times, times[valid_rate_mask], rpm_rate[valid_rate_mask])
    
    return times, rpms_plot, rpm_rate_plot, rpms

def plot_data(times: np.ndarray, rpms: np.ndarray, rpm_rate: np.ndarray, title: str, output: str):
    plt.rcParams['font.family'] = 'monospace'
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), facecolor=BG)
    fig.suptitle(title, color=FG, fontsize=14)
    
    # Fan speed over time
    ax1.plot(times * 1000, rpms, color=FG, linewidth=1)
    ax1.set_xlabel("Time (mS)", color=FG)
    ax1.set_ylabel("RPM", color=FG)
    ax1.set_title("Fan Speed", color=FG)
    ax1.set_facecolor(BG)
    ax1.tick_params(colors=FG)
    ax1.grid(True, color=FG, alpha=0.3)
    ax1.spines['bottom'].set_color(FG)
    ax1.spines['top'].set_color(FG)
    ax1.spines['left'].set_color(FG)
    ax1.spines['right'].set_color(FG)
    
    # Rate of change
    ax2.plot(times * 1000, rpm_rate, color=FG, linewidth=1)
    ax2.set_xlabel("Time (mS)", color=FG)
    ax2.set_ylabel("RPM/s", color=FG)
    ax2.set_title("Rate of Change", color=FG)
    ax2.set_facecolor(BG)
    ax2.tick_params(colors=FG)
    ax2.grid(True, color=FG, alpha=0.3)
    ax2.spines['bottom'].set_color(FG)
    ax2.spines['top'].set_color(FG)
    ax2.spines['left'].set_color(FG)
    ax2.spines['right'].set_color(FG)
    
    plt.tight_layout()
    plt.savefig(output, facecolor=BG, dpi=150)
    plt.close()

def main():
    files = [
        ("../data/fan_const_rpm.dat", "Constant RPM", 1100, None, 10),
        ("../data/fan_varying_rpm.dat", "Varying RPM", 1100, 1300, 20),
        ("../data/fan_varying_rpm_turning.dat", "Varying RPM (Turning)", 1100, 1300, 25),
    ]
    
    for dat_path, title, expected_min, expected_max, expected_duration in files:
        times, rpms_plot, rpm_rate_plot, rpms_original = process_file(dat_path)
        output = f"visualization_{dat_path.split('/')[-1].replace('.dat', '')}.png"
        plot_data(times, rpms_plot, rpm_rate_plot, title, output)
        
        # Diagnostic: verify alignment with expected ranges (use original, non-interpolated values)
        valid_rpms = rpms_original[~np.isnan(rpms_original)]
        if len(valid_rpms) > 0:
            duration_s = times[-1] - times[0] if len(times) > 0 else 0
            rpm_min = np.min(valid_rpms)
            rpm_max = np.max(valid_rpms)
            rpm_mean = np.mean(valid_rpms)
            rpm_std = np.std(valid_rpms)
            
            print(f"\n{title}:")
            print(f"  Duration: {duration_s:.1f}s (expected: ~{expected_duration}s)")
            print(f"  RPM range: {rpm_min:.1f} - {rpm_max:.1f} (mean: {rpm_mean:.1f}, std: {rpm_std:.1f})")
            if expected_max is None:
                print(f"  Expected: ~{expected_min} RPM constant")
                print(f"  Alignment: {'PASS' if abs(rpm_mean - expected_min) < 100 and rpm_std < 50 else 'CHECK'}")
            else:
                print(f"  Expected: {expected_min}-{expected_max} RPM")
                in_range = (rpm_min >= expected_min - 50) and (rpm_max <= expected_max + 50)
                print(f"  Alignment: {'PASS' if in_range else 'CHECK'}")
        print(f"  Saved: {output}")

if __name__ == "__main__":
    main()

