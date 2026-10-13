"""Validation of operator-supplied directories (DuckDB spill directory, archive targets).

These paths come from configuration or the CLI and decide where patient-level derived data may be written,
so they are checked before anything is created in them:

- absolute, with no '..' components, and free of symbolic links (the resolved path equals the given path),
  so a link planted in a shared directory cannot redirect writes;
- optionally confined below an allowed root;
- an existing directory must be a directory owned by the current user and not writable by others;
- missing directories are created with mode 0700.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path


class UnsafePathError(ValueError):
    """A configured directory is not acceptable; the message names the setting and the problem."""


def safe_directory(path: Path | str, purpose: str, root: Path | None = None, create: bool = True) -> Path:
    p = Path(path)
    if not p.is_absolute():
        raise UnsafePathError(f"{purpose}: must be an absolute path")
    if ".." in p.parts:
        raise UnsafePathError(f"{purpose}: must not contain '..'")
    norm = Path(os.path.normpath(p))
    if root is not None:
        r = Path(root)
        if not r.is_absolute():
            raise UnsafePathError(f"{purpose}: the allowed root must be an absolute path")
        if Path(os.path.realpath(r)) != Path(os.path.normpath(r)):
            raise UnsafePathError(f"{purpose}: the allowed root must not contain symbolic links")
        if norm != r and r not in norm.parents:
            raise UnsafePathError(f"{purpose}: must be inside the allowed root directory")
    # every existing component must be a real directory, never a symbolic link
    for part in [*reversed(norm.parents), norm]:
        if part.is_symlink():
            raise UnsafePathError(f"{purpose}: must not contain symbolic links")
    if not norm.exists():
        if not create:
            raise UnsafePathError(f"{purpose}: directory does not exist")
        try:
            norm.mkdir(mode=0o700, parents=True)
        except OSError as exc:
            raise UnsafePathError(f"{purpose}: cannot create directory ({exc.strerror})") from exc
    st = norm.lstat()
    if not stat.S_ISDIR(st.st_mode):
        raise UnsafePathError(f"{purpose}: not a directory")
    if hasattr(os, "getuid") and st.st_uid != os.getuid():
        raise UnsafePathError(f"{purpose}: directory must be owned by the user running the service")
    if st.st_mode & stat.S_IWOTH:
        raise UnsafePathError(f"{purpose}: directory must not be writable by other users")
    if not os.access(norm, os.W_OK | os.X_OK):
        raise UnsafePathError(f"{purpose}: directory is not writable")
    if Path(os.path.realpath(norm)) != norm:
        raise UnsafePathError(f"{purpose}: must not contain symbolic links")
    return norm
