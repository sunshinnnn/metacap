"""
@Author: Guoxing Sun
@Email: gsun@mpi-inf.mpg.de
@Date: 2022-12-25
"""

import sys
import os
import cv2
import numpy as np
from sys import platform
from matplotlib import pyplot as plt
import logging
import time
import datetime

# if platform == "win32" or platform=="win64":
import torch
to_cpu = lambda tensor: tensor.detach().cpu().numpy()
to_tensor = lambda numpy: torch.Tensor(numpy)
to_tensorFloat = lambda numpy: torch.Tensor(numpy).float()

# import smplx
import json
import shutil

def copyFile(pathIn, pathOut, fileNameIn, fileNameOut=None):
    if fileNameOut is None:
        fileNameOut = fileNameIn
    src = os.path.join(pathIn, fileNameIn)
    dst = os.path.join(pathOut, fileNameOut)
    shutil.copyfile(src, dst)

def saveNpy(path, data):
    np.save(path, data)

def saveJson(path, data):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)


def addText(img, text='0', bottomLeftCornerOfText = (40,40), fontScale = 2, fontColor              = (255,255,255),     thickness              = 2):
    if not isinstance(text, str):
        text = str(text)
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    # bottomLeftCornerOfText = (40,40)  #(10,500)
    # fontScale              = 2
    # fontColor              = (255,255,255)
    thickness              = 2
    lineType               = 2

    cv2.putText(img, text,
            bottomLeftCornerOfText,
            font,
            fontScale,
            fontColor,
            thickness,
            lineType)
    # return img

def grad_require(paras, flag=False):
    if isinstance(paras, list):
        for par in paras:
            par.requires_grad = flag
    elif isinstance(paras, dict):
        for key, par in paras.items():
            par.requires_grad = flag

def get_eta_str(cur_iter, total_iter, time_per_iter):
    eta = time_per_iter * (total_iter - cur_iter - 1)
    return str(datetime.timedelta(seconds=round(eta)))

def makePath(*args, **kwargs):
    '''
    if the path does not exist make it
    :param desired_path: can be path to a file or a folder name
    :return:
    '''
    isfile = kwargs.get('isfile', False)
    import os
    desired_path = os.path.join(*args)
    if isfile:
        if not os.path.exists(os.path.dirname(desired_path)):os.makedirs(os.path.dirname(desired_path, exist_ok=True))
    else:
        if not os.path.exists(desired_path): os.makedirs(desired_path,exist_ok=True)
    return desired_path


def makeSquareImg(img):
    h, w = img.shape[:2]
    if h==w:
        pass
    elif h>w:
        padSize0 = int((h-w)/2)
        padSize1 = int((h-w+1)/2)
        img = np.pad(img, ((0,0), (padSize0, padSize1), (0,0)),'constant', constant_values=(0, 0))
    elif h<w:
        padSize0 = int((w-h)/2)
        padSize1 = int((w-h+1)/2)
        img = np.pad(img, ((padSize0,padSize1),(0,0),(0,0)),'constant', constant_values=(0, 0))
    assert img.shape[0] == img.shape[1]
    return img

def resizeImg(img, scale=1.0, h=-1, w=-1,  interType=None):
    # scale_percent = scale       # percent of original size
    # width = int(img.shape[1] * scale_percent / 100)
    # height = int(img.shape[0] * scale_percent / 100)
    width = int(img.shape[1] * scale )
    height = int(img.shape[0] * scale )
    if h>0 and w>0:
        width = w
        height = h
    if interType =='nearest':
        inter_type = cv2.INTER_NEAREST
    elif interType =='area':
        inter_type = cv2.INTER_AREA
    elif interType =='linear':
        inter_type = cv2.INTER_LINEAR
    elif interType == 'cubic':
        inter_type = cv2.INTER_CUBIC
    else:
        inter_type = cv2.INTER_AREA
    img = cv2.resize(img, (width, height), interpolation = inter_type)
    return img


def checkPlatformDir(path):
    if path == '' or path is None:
        return None
    if platform == "win32" or platform=="win64":
        win = True
    else:
        win = False
    if not win:
        if platform == "linux" or platform == "linux2":
            if path[:2]=='Z:':
                path = '/HPS'+ path[2:]
            elif path[:2]=='Y:':
                path = '/CT'+ path[2:]
            else:
                pass
    else:
        if platform == "win32":
            if path[:3] == '/HP':
                path = 'Z:' + path[4:]
            elif path[:3] == '/CT':
                path = 'Y:' + path[3:]
            else:
                pass
    return path

def mask_to_rect(image, output=None):
    ret,thresh = cv2.threshold(image, 0, 1, 0)
    if (int(cv2.__version__[0]) > 3):
        contours, hierarchy = cv2.findContours(thresh.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    else:
        im2, contours, hierarchy = cv2.findContours(thresh.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
    # Getting the biggest contour
    if len(contours) != 0:
        # draw in blue the contours that were founded
        if not output is None:
            cv2.drawContours(output, contours, -1, 255, 3)
        # find the biggest countour (c) by the area
        c = max(contours, key = cv2.contourArea)
        x,y,w,h = cv2.boundingRect(c) 
    return [x, y, w, h]

def filter_mask(mask):
    if len(mask.shape)==3:
        x,y,w,h = mask_to_rect(mask[:,:,0])
    else:
        x, y, w, h = mask_to_rect(mask[:, :])
    maskOut = np.zeros_like(mask)
    maskOut[y:y+h,x:x+w] = mask[y:y+h,x:x+w]
    return maskOut

def get_palette(num_cls):
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    palette = np.array(palette).reshape(-1,3)
    return palette

def image_grid(images, rows=None, cols=None,
               dpi=100, fill: bool = True, show_axes: bool = False, rgb: bool = True,
               ):
    if (rows is None) != (cols is None):
        raise ValueError("Specify either both rows and cols or neither.")

    if rows is None:
        rows = len(images)
        cols = 1
    plt.rcParams['figure.dpi'] = dpi

    if rows == 1 and cols == 1:
        plt.imshow(images[0])
    else:
        gridspec_kw = {"wspace": 0.0, "hspace": 0.0} if fill else {}
        fig, axarr = plt.subplots(rows, cols, gridspec_kw=gridspec_kw, figsize=(15, 9))
        bleed = 0
        fig.subplots_adjust(left=bleed, bottom=bleed, right=(1 - bleed), top=(1 - bleed))

        for ax, im in zip(axarr.ravel(), images):
            if rgb:
                # only render RGB channels
                ax.imshow(im[..., :3])
            else:
                # only render Alpha channel
                ax.imshow(im[...])
            if not show_axes:
                ax.set_axis_off()


FORMAT_DICTS={
    'tm': "%(asctime)s: %(message)s",
    'tlm': "%(asctime)s: [%(filename)s:%(funcName)s:%(lineno)d] %(message)s",
    'lm': "[%(filename)s:%(funcName)s:%(lineno)d] %(message)s",
    'm': "%(message)s",
}
# FORMAT = "%(asctime)s [%(filename)s:%(funcName)s:%(lineno)d] %(message)s"

def setup_logger(save_dir='', log_filename=None, log=True, save = True, format_type='m'):
    # if not os.path.exists(save_dir):
    #     print('[ERROR] You must give a dir path to save logger.')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # formatter = logging.Formatter("%(asctime)s: %(message)s")
    # formatter = logging.Formatter("%(message)s")
    formatter = logging.Formatter(FORMAT_DICTS[format_type])
    if log_filename is None:
        log_filename = os.path.join(save_dir, time.strftime("%Y-%m-%d_%H:%M:%S@", time.localtime()).replace(':','~') +'logging.txt')
    else:
        log_filename = os.path.join(save_dir, log_filename + 'logging.txt')
    if save:
        os.makedirs(save_dir, exist_ok=True)
        fh = logging.FileHandler(log_filename, mode='w')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    if log:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger, log_filename

def clip_min(t, t_min):
    """
    clip_by_tensor
    :param t: tensor
    :param t_min: min
    :param t_max: max
    :return: cliped tensor
    """
    t = t.float()
    # t_min=t_min.float()
    # t_max=t_max.float()

    result = (t >= t_min).float() * t + (t < t_min).float() * t_min
    # result = (result <= t_max).float() * result + (result > t_max).float() * t_max
    return result