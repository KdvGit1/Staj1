"""Low-buffer RTSP reader that exposes only the newest decoded frame."""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from urllib.parse import quote


def build_hikvision_rtsp_url(
    ip: str,
    username: str,
    password: str,
    channel: str = "202",
    rtsp_port: int = 554,
) -> str:
    """Build a credential-safe Hikvision RTSP URL without logging it."""
    if not ip or not username or not password:
        raise ValueError("Camera IP, username and password are required.")
    user = quote(username, safe="")
    secret = quote(password, safe="")
    return (
        f"rtsp://{user}:{secret}@{ip}:{rtsp_port}"
        f"/Streaming/Channels/{channel}"
    )


@dataclass(frozen=True)
class StreamFrame:
    sequence: int
    received_monotonic: float
    image: object


class LatestFrameStream:
    """Continuously decode a stream and discard frames superseded by newer ones."""

    def __init__(
        self,
        source: str | int,
        reconnect: bool = True,
        reconnect_delay: float = 1.0,
    ) -> None:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError(
                "OpenCV is required for camera capture. Install "
                "thermal_live_detection/requirements.txt."
            ) from exc

        os.environ.setdefault(
            "OPENCV_FFMPEG_CAPTURE_OPTIONS",
            (
                "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|"
                "buffer_size;1024000|reorder_queue_size;0"
            ),
        )
        self._cv2 = cv2
        self.source = source
        self.reconnect = reconnect
        self.reconnect_delay = reconnect_delay
        self._capture = None
        self._condition = threading.Condition()
        self._latest: StreamFrame | None = None
        self._sequence = 0
        self._stopping = False
        self._thread: threading.Thread | None = None

    def _open(self) -> bool:
        self._release()
        backend = (
            self._cv2.CAP_FFMPEG
            if isinstance(self.source, str)
            else self._cv2.CAP_ANY
        )
        capture = self._cv2.VideoCapture(self.source, backend)
        capture.set(self._cv2.CAP_PROP_BUFFERSIZE, 1)
        if not capture.isOpened():
            capture.release()
            return False
        self._capture = capture
        return True

    def _release(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def start(self) -> "LatestFrameStream":
        if self._thread is not None:
            return self
        self._thread = threading.Thread(
            target=self._run,
            name="thermal-rtsp-reader",
            daemon=True,
        )
        self._thread.start()
        return self

    def _run(self) -> None:
        consecutive_failures = 0
        while not self._stopping:
            if self._capture is None and not self._open():
                if not self.reconnect:
                    break
                time.sleep(self.reconnect_delay)
                continue

            ok, frame = self._capture.read()
            if not ok or frame is None:
                consecutive_failures += 1
                if consecutive_failures >= 10:
                    self._release()
                    consecutive_failures = 0
                    if not self.reconnect:
                        break
                    time.sleep(self.reconnect_delay)
                continue

            consecutive_failures = 0
            with self._condition:
                self._sequence += 1
                self._latest = StreamFrame(
                    sequence=self._sequence,
                    received_monotonic=time.monotonic(),
                    image=frame,
                )
                self._condition.notify_all()

        self._release()
        with self._condition:
            self._condition.notify_all()

    def read(
        self,
        after_sequence: int = 0,
        timeout: float = 2.0,
    ) -> StreamFrame | None:
        """Wait for a frame newer than ``after_sequence``."""
        deadline = time.monotonic() + timeout
        with self._condition:
            while not self._stopping:
                if (
                    self._latest is not None
                    and self._latest.sequence > after_sequence
                ):
                    return StreamFrame(
                        sequence=self._latest.sequence,
                        received_monotonic=self._latest.received_monotonic,
                        image=self._latest.image.copy(),
                    )
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._condition.wait(remaining)
        return None

    def stop(self) -> None:
        self._stopping = True
        with self._condition:
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._release()

    def __enter__(self) -> "LatestFrameStream":
        return self.start()

    def __exit__(self, *_: object) -> None:
        self.stop()
