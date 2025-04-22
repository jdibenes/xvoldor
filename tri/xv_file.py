
import os


def scan_files(path):
    items = os.listdir(path)
    paths = [os.path.join(path, item) for item in items]
    files = [path for path in paths if (os.path.isfile(path))]
    return files


def get_file_name(path):
    root, tail = os.path.split(path)
    name, ext = os.path.splitext(tail)
    return (root, name, ext)


class image_loader:
    pass

