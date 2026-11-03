"""ROS-free NaVILA eight-frame history.

Ported from the authoritative
``NaVILA/integrations/arena/history.py``.  Samples are accumulated at 0.5 s
intervals, the latest observation is always the final frame, and missing early
frames are filled with same-size BLACK images (never by repeating the first
frame).
"""
from __future__ import annotations

import base64
import io


def sample_indices(count: int) -> list[int | None]:
    if count < 1:
        raise ValueError('at least one image is required')
    padded = max(8, count)
    pad = padded - count
    return [
        None if i < pad else i - pad
        for i in [int(i * (padded - 1) / 7) for i in range(7)] + [padded - 1]
    ]


def encode_image(image) -> str:
    buffer = io.BytesIO()
    image.convert('RGB').save(buffer, format='JPEG')
    return base64.b64encode(buffer.getvalue()).decode('ascii')


class FrameHistory:
    """0.5 s sampling history with duplicate rejection and black padding."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.last_seen = None
        self.next_sample = None
        self.frames: list[tuple[int, str]] = []
        self.latest = None
        self.duplicates = 0

    def offer(self, stamp: int, image) -> bool:
        stamp = int(stamp)
        if self.last_seen is not None and stamp <= self.last_seen:
            self.duplicates += 1
            return False
        self.last_seen = stamp
        self.latest = (stamp, image.copy())
        if self.next_sample is None or stamp >= self.next_sample:
            self.frames.append((stamp, encode_image(image)))
            self.next_sample = stamp + 500_000_000
        return True

    def sample(self) -> tuple[list[str], list[int | None]]:
        from PIL import Image

        if self.latest is None:
            raise ValueError('no current image')
        current, image = self.latest
        frames = list(self.frames)
        if not frames or frames[-1][0] != current:
            frames.append((current, encode_image(image)))
        black = encode_image(Image.new('RGB', image.size, (0, 0, 0)))
        selected = [frames[i] if i is not None else (None, black) for i in sample_indices(len(frames))]
        return [item[1] for item in selected], [item[0] for item in selected]
