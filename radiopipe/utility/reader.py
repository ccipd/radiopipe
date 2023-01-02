from os import getenv
from pathlib import Path

from .custom_type import PathType

root_in = getenv("ROOT_IN") or "."
root_out = getenv("ROOT_OUT") or "."


def path_in(p: str) -> PathType:
    path: PathType = Path(rf"{root_in}/{p}")
    if not path.exists():
        raise Exception(f"{p} doesn't exist on the earth.")

    return path


def path_out(p: str) -> PathType:
    path: PathType = Path(rf"{root_out}/{p}")
    if not path.exists():
        print("The output path doesn't exist but no worries I'm making it.")
        path.mkdir(parents=True)

    return path
