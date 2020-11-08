import glob
import os

root_path = ''

_ALLOWED_IMAGE_FORMATS = ('.jpg', '.jpeg', '.png')


def fileName(file_path, ext=False):
    head, tail = os.path.split(file_path)
    if ext is False:
        file_name = '.'.join(tail.split('.')[:-1])
    else:
        file_name = tail
    return file_name


def find_images(path):

    res = list()
    res = [
        f for f in glob.glob(os.path.join(path, '*.*'))
        if f.endswith(_ALLOWED_IMAGE_FORMATS)
    ]
    return res
