from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Tuple

import numpy as np


@dataclass
class RpmEstimate:
    timestamp_s: float
    rpm: Optional[float]


class RpmEstimator:
    """
    Estimate rotational speed (RPM) from event-count oscillations.

    Approach:
    - Track the total event count per window vs. time.
    - Resample to a uniform grid and compute rFFT.
    - Peak-pick dominant frequency within [min_hz, max_hz], convert to RPM.
    """

    def __init__(
        self,
        *,
        min_hz: float = 5.0,
        max_hz: float = 80.0,
        history_s: float = 5.0,
        min_duration_s: float = 1.5,
    ) -> None:
        self.min_hz = float(min_hz)
        self.max_hz = float(max_hz)
        self.history_s = float(history_s)
        self.min_duration_s = float(min_duration_s)

        self._times: Deque[float] = deque()
        self._values: Deque[float] = deque()
        self._last_estimate: Optional[float] = None

    def update(
        self,
        window: Tuple[np.ndarray, np.ndarray, np.ndarray],
        t_end_s: float,
        window_duration_s: float,
    ) -> RpmEstimate:
        x, y, pol = window  # noqa: ARG002 - x,y not used directly

        # Use total activity as proxy signal for rotation-induced modulation.
        total_events = float(x.size)

        # Append new sample at the end timestamp of the window.
        self._times.append(float(t_end_s))
        self._values.append(total_events)

        # Trim old history.
        cutoff = t_end_s - self.history_s
        while self._times and self._times[0] < cutoff:
            self._times.popleft()
            self._values.popleft()

        rpm = self._estimate_from_history(window_duration_s)
        self._last_estimate = rpm if rpm is not None else self._last_estimate
        return RpmEstimate(timestamp_s=t_end_s, rpm=self._last_estimate)

    def _estimate_from_history(self, window_duration_s: float) -> Optional[float]:
        if len(self._times) < 4:
            return None

        t0 = self._times[0]
        t1 = self._times[-1]
        duration = t1 - t0
        if duration < self.min_duration_s:
            return None

        # Resample to uniform grid at approximately the native window rate.
        # Guard against zero durations.
        if window_duration_s <= 0:
            return None

        # Use a conservative oversampling factor to stabilize FFT.
        nominal_fs = max(1.0 / window_duration_s, 20.0)
        fs = float(nominal_fs)

        n = int(np.clip(int(duration * fs), 64, 4096))
        t_uniform = np.linspace(t0, t1, n, dtype=np.float64)

        times = np.fromiter(self._times, dtype=np.float64, count=len(self._times))
        values = np.fromiter(self._values, dtype=np.float64, count=len(self._values))

        # Interpolate to uniform grid.
        y_uniform = np.interp(t_uniform, times, values)

        # Detrend and windowing to reduce spectral leakage.
        y_uniform = y_uniform - y_uniform.mean()
        if np.allclose(y_uniform, 0.0):
            return None
        window = np.hanning(n)
        y_win = y_uniform * window

        # Real FFT.
        spec = np.fft.rfft(y_win)
        freqs = np.fft.rfftfreq(n, d=1.0 / fs)
        mag = np.abs(spec)

        # Focus on plausible band.
        band = (freqs >= self.min_hz) & (freqs <= self.max_hz)
        if not np.any(band):
            return None

        idx = np.argmax(mag[band])
        peak_hz = float(freqs[band][idx])

        # Convert to RPM.
        rpm = peak_hz * 60.0

        # Simple plausibility clamp to avoid wild swings.
        if rpm <= 0 or not np.isfinite(rpm):
            return None
        return rpm

    @staticmethod
    def overlay_text(
        frame: np.ndarray,
        text: str,
        origin: Tuple[int, int] = (8, 60),
        color: Tuple[int, int, int] = (0, 0, 0),
    ) -> None:
        import cv2  # Local import to avoid hard dependency at import-time

        cv2.putText(
            frame,
            text,
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

