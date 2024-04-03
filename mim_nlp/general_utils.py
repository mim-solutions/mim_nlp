import os
from pathlib import Path
from typing import Union


def get_size_in_megabytes(start_path: Union[str, Path] = ".") -> str:
    start_path = Path(start_path)
    total_size = 0
    for dirpath, _, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)

    size_in_mb = total_size >> 20
    if size_in_mb != 0:
        return f"{size_in_mb} MB"
    else:
        return f"{total_size >> 10} KB"
