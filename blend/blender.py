import os, cv2
import numpy as np

from utils import read_img, save_img
from blend.laplacian_pyramid import LaplacianPyramid as LP

class LaplacianPyramid(object):

    @staticmethod
    def downSamplePyramids(image, n_level, sigma=1):
        pyramids = [image]
        for i in range(1, n_level):
            temp = pyramids[i - 1].copy()
            rows, cols, _ = temp.shape
            temp = temp[range(0, rows, 2), :, :]
            temp = temp[:, range(0, cols, 2), :]
            temp = cv2.GaussianBlur(temp, (5, 5), sigma)
            pyramids.append(temp)
        return pyramids

    @staticmethod
    def upSample(image):
        rows, cols, channels = image.shape
        up_image = np.zeros((rows * 2, cols, channels))
        up_image[range(0, rows * 2, 2), :, :] = image
        up_image[range(1, rows * 2, 2), :, :] = image
        up_image2 = np.zeros((rows * 2, cols * 2, channels))
        up_image2[:, range(0, cols * 2, 2), :] = up_image
        up_image2[:, range(1, cols * 2, 2), :] = up_image
        return up_image2

    @staticmethod
    def buildLaplacianPyramids(image, n_level):
        h_filter = np.reshape(np.array([1, 4, 6, 4, 1]) / 16.0, [5, 1])
        h_filter = np.matmul(h_filter, h_filter.transpose())
        g_filter = h_filter

        pyramids = []
        cur_image = image
        for i in range(n_level - 1):
            temp = cv2.filter2D(cur_image, -1, h_filter)
            rows, cols, _ = temp.shape
            temp = temp[range(0, rows, 2), :, :]
            temp = temp[:, range(0, cols, 2), :]
            dn_temp = temp.copy()
            temp = LaplacianPyramid.upSample(temp)
            temp = temp[:rows, :cols, :]
            temp = cv2.filter2D(temp, -1, g_filter)
            pyramids.append(cur_image - temp)
            cur_image = dn_temp
        pyramids.append(cur_image)
        return pyramids

    @staticmethod
    def reconstruct(pyramids):
        h_filter = np.reshape(np.array([1, 4, 6, 4, 1]) / 16.0, [5, 1])
        h_filter = np.matmul(h_filter, h_filter.transpose())
        g_filter = h_filter

        for i in range(len(pyramids) - 1, 0, -1):
            temp = pyramids[i]
            temp = LaplacianPyramid.upSample(temp)
            rows, cols, _ = pyramids[i - 1].shape
            temp = temp[:rows, :cols, :]
            temp = cv2.filter2D(temp, -1, g_filter)
            pyramids[i - 1] = pyramids[i - 1] + temp
        return pyramids[0]

def laplacian_pyramid_blend(template_tex, input_tex, mask, times=5):
    '''
    Blend using Laplacian Pyramid.

    Args:
        template_tex: numpy.array (unwrap_size, unwrap_size, 3). The template texture.
        input_tex: numpy.array (unwrap_size, unwrap_size, 3). The input texture.
        mask: numpy.array (unwrap_size, unwrap_size, 3). The mask of input texture.
        times: numpy.array (unwrap_size, unwrap_size, 3). The number of Laplacian Pyramid levels.
    Returns:
        blend_tex: numpy.array (unwrap_size, unwrap_size, 3). The blended texture.
    '''
    pyramids_template = LP.buildLaplacianPyramids(template_tex, times)
    pyramids_input = LP.buildLaplacianPyramids(input_tex, times)
    mask_list = LP.downSamplePyramids(mask, times)
    pyramids_blend = []
    for i in range(len(pyramids_template)):
        mask = np.clip(mask_list[i], 0, 1)
        pyramids_blend.append(pyramids_input[i] * mask + pyramids_template[i] * (1 - mask))
    blend_tex = LP.reconstruct(pyramids_blend)
    blend_tex = np.clip(blend_tex, 0, 255)
    return blend_tex

def blending(args, unwrap_size):
    for fn in args.input_fnames:
        # 1. create template tex
        fn_right_tex = os.path.splitext(fn)[0] + '_right_tex.png'
        right_tex_path = os.path.join('unwrap/results', fn_right_tex)
        fn_left_tex = os.path.splitext(fn)[0] + '_left_tex.png'
        left_tex_path = os.path.join('unwrap/results', fn_left_tex)
        left_mask_path = 'blend/mask/left_mask.png'

        right_tex = read_img(right_tex_path, resize=(unwrap_size, unwrap_size))
        left_tex = read_img(left_tex_path, resize=(unwrap_size, unwrap_size))
        left_mask = read_img(left_mask_path, resize=(unwrap_size, unwrap_size), dst_range=1.)
        
        template_tex = laplacian_pyramid_blend(template_tex=right_tex, input_tex=left_tex, mask=left_mask)

        # 2. create final tex
        fn_front_tex = os.path.splitext(fn)[0] + '_tex.png'
        front_tex_path = os.path.join('unwrap/results', fn_front_tex)
        front_mask_path = 'blend/mask/front_mask.png'
        
        front_tex = read_img(front_tex_path, resize=(unwrap_size, unwrap_size))
        front_mask = read_img(front_mask_path, resize=(unwrap_size, unwrap_size), dst_range=1.)
        
        final_tex = laplacian_pyramid_blend(template_tex=template_tex, input_tex=front_tex, mask=front_mask)
        
        dirname = os.path.splitext(fn)[0]
        result_dir = os.path.join(args.output_dir, dirname)
        save_img(final_tex, os.path.join(result_dir, 'texture.png'))