import matplotlib.pylab as plt
import numpy as np
from PIL import Image

def show_img(img, title):
    plt.imshow(img)
    plt.title(title)
    plt.axis('off')

def show_result(src, dst, mask, result):
    plt.figure(figsize=(20, 20))
    
    plt.subplot(1,4,1)
    show_img(np2pillow(src, 255), 'src')

    plt.subplot(1,4,2)
    show_img(mask, 'mask')

    plt.subplot(1,4,3)
    show_img(np2pillow(dst, 255), 'dst')

    plt.subplot(1,4,4)
    show_img(np2pillow(result, 255), 'result')
    
    plt.show()

def pillow2np(img, dst_range=255.):
    coef = dst_range / 255.
    return np.asarray(img, np.float32) * coef

def read_img(path, resize=None, dst_range=255.):
    img = Image.open(path)

    if resize is not None:
        img = img.resize(resize)

    img = pillow2np(img, dst_range)

    if img.ndim == 2:
        img = np.tile(img[:, :, np.newaxis], (1, 1, 3))
    if img.shape[2] == 1:
        img = np.tile(img, (1, 1, 3))
    if img.shape[2] > 3:
        img = img[:, :, :3]

    return img

def np2pillow(img, src_range=255.):
    coef = 255. / src_range
    return Image.fromarray(np.squeeze(np.clip(np.round(img * coef), 0, 255).astype(np.uint8)))

def save_img(img, path, src_range=255., mask=False):
    if mask == True:
        img.save(path)
    else:
        np2pillow(img, src_range).save(path)

def img3channel(img):
    '''make the img to have 3 channels'''
    if img.ndim == 2:
        img = np.tile(img[:, :, np.newaxis], (1, 1, 3))
    if img.shape[2] == 1:
        img = np.tile(img, (1, 1, 3))
    if img.shape[2] > 3:
        img = img[:, :, :3]
    return img

def img2mask(img, thre=128, mode='greater'):
    '''mode: greater/greater-equal/less/less-equal/equal'''
    if mode == 'greater':
        mask = (img > thre).astype(np.float32)
    elif mode == 'greater-equal':
        mask = (img >= thre).astype(np.float32)
    elif mode == 'less':
        mask = (img < thre).astype(np.float32)
    elif mode == 'less-equal':
        mask = (img <= thre).astype(np.float32)
    elif mode == 'equal':
        mask = (img == thre).astype(np.float32)
    else:
        raise NotImplementedError

    mask = img3channel(mask)

    return mask