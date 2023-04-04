import os
import cv2
import time
import numpy as np
import torch
from torch.autograd import Variable

from lights.models.defineHourglass_512_gray_skip import *
from utils.utils_SH import *


def get_light_vec(Lab, network, device):

    inputL = Lab[:, :, 0]
    inputL = inputL.astype(np.float32) / 255.0
    inputL = inputL.transpose((0, 1))
    inputL = inputL[None, None, ...]
    inputL = Variable(torch.from_numpy(inputL).to(device))

    # we only need the encoder
    # target_sh is for decoder
    target_sh = np.array([[
        [[0.8642]],
        [[-0.1094]],
        [[0.2552]],
        [[-0.1115]],
        [[-0.0581]],
        [[-0.0465]],
        [[-0.2389]],
        [[-0.0134]],
        [[0.1331]],
    ]]).astype(np.float32)
    sh = Variable(torch.from_numpy(target_sh).to(device))
    _, light = network(inputL, sh, 0)

    return light


def det_attributes(args):
    '''Usage
    cd ./DataSet_Step2_Det_Attributes
    python run_dpr_light.py \
        --proj_data_dir ../examples/dataset_examples \
        --dpr_model_path ../checkpoints/dpr_model/trained_model_03.t7
    '''

    # ----------------------- Define Inference Parameters -----------------------
    model_path = args.dpr_model_path
    input_images_dir = 'inversion/results'
    output_dir = 'lights/results'
    os.makedirs(output_dir, exist_ok=True)
    device = args.device

    # ----------------------- Load model -----------------------
    my_network = HourglassNet()
    my_network.load_state_dict(torch.load(model_path))
    print('DPR model successfully loaded from {}!'.format(model_path))

    my_network.to(device)
    my_network.train(False)

    # ----------------------- Calcuate light -----------------------
    suffix = '_inverted'
    fnames = sorted(os.listdir(input_images_dir))
    for fn in fnames:
        basename = fn[:fn.rfind('.')]
        if basename.rfind(suffix) == -1:
            continue

        tic = time.time()

        img = cv2.imread(os.path.join(input_images_dir, fn))
        img = cv2.resize(img, (512, 512))
        Lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        lv = get_light_vec(Lab, my_network, device)
        np.save(os.path.join(output_dir, f'{basename[:-len(suffix)]}.npy'), lv.data.cpu().numpy())

        toc = time.time()
        print('Calculate light {} took {:.4f} seconds.'.format(fn, toc - tic))

    print('Calculate light Done!')
