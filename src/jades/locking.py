"""Non-blocking process locks, released by the OS even after process death."""
from contextlib import contextmanager
import hashlib
import os
from pathlib import Path


class OutputInUseError(RuntimeError):
    pass


@contextmanager
def output_lock(output):
    key = os.path.normcase(str(Path(output).resolve()))
    directory = Path(output).parent.resolve() / ".jades-locks"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (hashlib.sha256(os.fsencode(key)).hexdigest() + ".lock")
    # Never unlink lock files: replacing a locked inode would allow another writer.
    with path.open("a+b") as handle:
        if os.name == "nt":
            import msvcrt
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError:
                raise OutputInUseError(f"Another evaluation is writing to {output}") from None
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise OutputInUseError(f"Another evaluation is writing to {output}") from None
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)
