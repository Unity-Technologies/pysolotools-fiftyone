import Imath
import numpy as np
import OpenEXR as exr
from PIL import Image


def _normalize_array(in_array, extent_max=10, extent_min=-10):
    clipped = np.clip(in_array, extent_min, extent_max)

    a_max = max(1, np.max(clipped))
    a_min = min(0, np.min(clipped))
    diff = a_max - a_min
    return (clipped - a_min) / diff


def exr_to_numpy(exr_img):
    dw = exr_img.header()['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

    r_str = exr_img.channel('R', Imath.PixelType(Imath.PixelType.FLOAT))
    g_str = exr_img.channel('G', Imath.PixelType(Imath.PixelType.FLOAT))
    b_str = exr_img.channel('B', Imath.PixelType(Imath.PixelType.FLOAT))
    a_str = exr_img.channel('A', Imath.PixelType(Imath.PixelType.FLOAT))

    r = _normalize_array(np.fromstring(r_str, dtype=np.float32).reshape(size[1], -1))
    g = _normalize_array(np.fromstring(g_str, dtype=np.float32).reshape(size[1], -1))
    b = _normalize_array(np.fromstring(b_str, dtype=np.float32).reshape(size[1], -1))
    a = _normalize_array(np.fromstring(a_str, dtype=np.float32).reshape(size[1], -1))

    return np.stack([r, g, b, a], 2)


def exr_path_to_numpy(path):
    exr_file = exr.InputFile(path)
    return exr_to_numpy(exr_file)


def exr_to_png(exr_img):
    exr_array = exr_to_numpy(exr_img)
    img = Image.fromarray(np.uint8(exr_array * 255))
    return img


def exr_path_to_png(path):
    exr_file = exr.InputFile(path)
    return exr_to_png(exr_file)


def convert_exr_to_png(exr_path, png_path):
    png_img = exr_path_to_png(exr_path)
    png_img.save(png_path)
