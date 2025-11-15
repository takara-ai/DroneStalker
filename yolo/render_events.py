import argparse
import cv2
import numpy as np

from evio.core.pacer import Pacer
from evio.source.dat_file import BatchRange, DatFileSource


def get_window(
    event_words: np.ndarray,
    time_order: np.ndarray,
    win_start: int,
    win_stop: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    idx = time_order[win_start:win_stop]
    words = event_words[idx].astype(np.uint32, copy=False)

    x = (words & 0x3FFF).astype(np.int32, copy=False)
    y = ((words >> 14) & 0x3FFF).astype(np.int32, copy=False)
    pol = ((words >> 28) & 0xF) > 0

    return x, y, pol


def get_frame(
    window: tuple[np.ndarray, np.ndarray, np.ndarray],
    width: int = 1280,
    height: int = 720,
    *,
    base_color: tuple[int, int, int] = (52, 37, 30),     # #1e2534
    on_color: tuple[int, int, int] = (255, 255, 255),    # #ffffff
    off_color: tuple[int, int, int] = (172, 109, 57),    # #396dac
) -> np.ndarray:

    x, y, pol = window

    frame = np.empty((height, width, 3), np.uint8)
    frame[:] = base_color

    frame[y[pol], x[pol]] = on_color
    frame[y[~pol], x[~pol]] = off_color

    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dat", help="Path to .dat file")
    parser.add_argument("--window", type=float, default=10,
                        help="Window duration in ms")
    parser.add_argument("--speed", type=float, default=1,
                        help="Playback speed")
    parser.add_argument("--force-speed", action="store_true",
                        help="Force playback speed by dropping windows")
    parser.add_argument("--out", default="output.mp4",
                        help="Output video filename")
    args = parser.parse_args()

    width, height = 1280, 720

    src = DatFileSource(
        args.dat, width=width, height=height,
        window_length_us=args.window * 1000
    )

    pacer = Pacer(speed=args.speed, force_speed=args.force_speed)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.out, fourcc, 60, (width, height))
    if not out.isOpened():
        raise RuntimeError("VideoWriter failed (FFMPEG missing?)")

    for batch in pacer.pace(src.ranges()):

        win = get_window(
            src.event_words,
            src.order,
            batch.start,
            batch.stop,
        )

        frame = get_frame(win, width=width, height=height)

        out.write(frame)

    out.release()


if __name__ == "__main__":
    main()
