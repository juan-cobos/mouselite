"""Live inference from a webcam or any other `cv2.VideoCapture` source."""

import threading
from collections.abc import Iterator
from types import TracebackType

import cv2
import numpy as np
import supervision as sv

from mouselite.pipeline import Pipeline


class LiveStream:
    """Context manager that opens a capture device and yields processed frames.

    By default every frame is yielded, in order, so a model slower than the camera falls
    behind real time as frames queue up in the driver. With `latest=True` a thread keeps
    reading and only the newest frame is kept: each step takes it and skips any frames in
    between, so the stream stays current. Don't use it for files, where every frame is
    wanted.
    """

    def __init__(
        self,
        pipeline: Pipeline,
        source: int | str = 0,
        width: int | None = None,
        height: int | None = None,
        latest: bool = False,
    ):
        self.pipeline = pipeline
        self.source = source
        self.width = width
        self.height = height
        self.latest = latest
        self.capture: cv2.VideoCapture | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._ready = threading.Condition()
        self._slot: np.ndarray | None = None
        self._ended = False

    def __enter__(self) -> "LiveStream":
        capture = cv2.VideoCapture(self.source)
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"could not open video source {self.source!r}")
        if self.width is not None:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height is not None:
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.capture = capture
        self.pipeline.reset()
        if self.latest:
            self._stop.clear()
            self._slot = None
            self._ended = False
            self._thread = threading.Thread(target=self._read_latest, daemon=True)
            self._thread.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    def _read_latest(self) -> None:
        """Reader thread: overwrite the slot with each new frame until stopped."""
        while not self._stop.is_set():
            ok, frame = self.capture.read()
            with self._ready:
                if not ok:
                    self._ended = True
                else:
                    self._slot = frame
                self._ready.notify_all()
            if not ok:
                return

    def _next_latest(self) -> np.ndarray | None:
        """Block for the newest frame not yet taken; None once the source ended."""
        with self._ready:
            self._ready.wait_for(lambda: self._slot is not None or self._ended)
            frame, self._slot = self._slot, None
            return frame

    def __iter__(self) -> Iterator[tuple[np.ndarray, sv.Detections]]:
        if self.capture is None:
            raise RuntimeError("LiveStream must be used as a context manager")
        frame_idx = 0
        while True:
            if self.latest:
                frame = self._next_latest()
            else:
                ok, frame = self.capture.read()
                frame = frame if ok else None
            if frame is None:
                return
            yield frame, self.pipeline.process_frame(frame, frame_idx)
            frame_idx += 1
