import os
import time
import argparse
import dlib
import torch
import torchvision.transforms as transforms

from utils.common import tensor2im
from utils.alignment import align_face
from inversion.models.psp import pSp  # we use the pSp framework to load the e4e encoder.


def run_alignment(image_path, shape_predictor_model_path):
    predictor = dlib.shape_predictor(shape_predictor_model_path)
    aligned_image = align_face(filepath=image_path, predictor=predictor)
    return aligned_image


def det_num_faces(image_path):
    detector = dlib.get_frontal_face_detector()
    img = dlib.load_rgb_image(image_path)
    dets = detector(img, 1)
    return len(dets)


def inversion(args):
    # ----------------------- Define Inference Parameters -----------------------
    # output_latents_dir = 'latents'
    # output_inversions_dir = 'inversions'
    # os.makedirs(output_latents_dir, exist_ok=True)
    # os.makedirs(output_inversions_dir, exist_ok=True)
    result_dir = 'inversion/results'
    os.makedirs(result_dir, exist_ok=True)

    # ----------------------- Load Pretrained Model -----------------------
    ckpt = torch.load(args.e4e_model_path, map_location='cpu')
    opts = ckpt['opts']
    opts['checkpoint_path'] = args.e4e_model_path
    opts = argparse.Namespace(**opts)
    net = pSp(opts).eval().to(args.device)

    # ----------------------- Setup Data Transformations -----------------------
    image_transformer = transforms.Compose(
        [transforms.Resize((256, 256)),
         transforms.ToTensor(),
         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]) # --> ~1 <= pix_val <= 1

    # ----------------------- Perform Inversion -----------------------
    for fn in args.input_fnames:
        basename = fn[:fn.rfind('.')]
        #print("fn: ", fn)
        tic = time.time()
        
        # detect if the image contains faces
        if det_num_faces(os.path.join(args.input_dir, fn)) < 1:
            print(f'>> Skip {fn}, there is no face detected in this image!')
            continue

        # align face --> 256x256, blurring background
        input_image = run_alignment(image_path=os.path.join(args.input_dir, fn),
                                    shape_predictor_model_path=args.shape_predictor_model_path)
        #input_image.save(os.path.join(result_dir, f"{basename}_aligned.png"))

        # preprocess image
        transformed_image = image_transformer(input_image)

        # inversion
        images, latents = net(transformed_image.unsqueeze(0).to(args.device).float(),
                              randomize_noise=False,
                              resize=False,
                              return_latents=True)

        torch.save(latents, os.path.join(result_dir, f"{basename}.pt"))
        tensor2im(images[0]).save(os.path.join(result_dir, f"{basename}_inverted.png"))
        
        if det_num_faces(os.path.join(result_dir, f"{basename}_inverted.png")) < 1:
            os.unlink(os.path.join(result_dir, f"{basename}.pt"))
            os.unlink(os.path.join(result_dir, f"{basename}_inverted.png"))
            print(f'>> Drop {fn}, there is no face detected in the inversed image!')

        toc = time.time()
        print('Inverse {} took {:.4f} seconds.'.format(fn, toc - tic))

    print('Inversion process done!')