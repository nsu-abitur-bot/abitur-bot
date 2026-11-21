"""Bounded cross-process transport gate on the same local budget volume."""

import asyncio
import os
import time
from pathlib import Path

from tools.admission_usage import IndexingLimit


class TransportGate:
    def __init__(self, budget: Path) -> None:
        self.path = Path(str(budget) + ".gate")
        self.file = None

    def attempt(self) -> bool:
        if self.file is None:
            if self.path.is_symlink():
                raise IndexingLimit("A symlink is not a pilot lock")
            self.file = self.path.open("a+b")
            if self.file.tell() == 0:
                self.file.write(b"0")
                self.file.flush()
            self.file.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self.file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except (OSError, BlockingIOError):
            return False

    def close(self) -> None:
        if self.file is not None:
            # Closing also releases the OS lock, including after process death.
            self.file.close()
            self.file = None

    def acquire(self) -> None:
        start = time.monotonic()
        try:
            while not self.attempt():
                if time.monotonic() - start > 180:
                    raise IndexingLimit("Pilot transport gate wait timed out")
                time.sleep(0.01)
        except BaseException:
            self.close()
            raise

    async def acquire_async(self) -> None:
        start = time.monotonic()
        try:
            while not self.attempt():
                if time.monotonic() - start > 180:
                    raise IndexingLimit("Pilot transport gate wait timed out")
                await asyncio.sleep(0.01)
        except BaseException:
            self.close()
            raise
