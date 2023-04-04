import os, time, torch, json
import numpy as np
import dnnlib
import dnnlib.tflib as tflib
from PIL import Image
from lights.dpr_light import det_attributes
from editing.pretrained_networks import load_networks
from editing.modules.flow import cnf
from editing.expression_recognition import Exp_Recog_API

ATTR_ID = {
    'Light': {
        'attr_idx': (0, 8),
        'w_idxs': [(7, 11)]
    },
    'Gender': {
        'attr_idx': 9,
        'w_idxs': [(0, 7)]
    },
    'Glasses': {
        'attr_idx': 10,
        'w_idxs': [(2, 3)]
    },
    'Yaw': {
        'attr_idx': 11,
        'w_idxs': [(0, 3)]
    },
    'Pitch': {
        'attr_idx': 12,
        'w_idxs': [(0, 3)]
    },
    'Baldness': {
        'attr_idx': 13,
        'w_idxs': [(0, 5)]
    },
    'Beard': {
        'attr_idx': 14,
        'w_idxs': [(5, 9)]
    },
    'Age': {
        'attr_idx': 15,
        'w_idxs': [(4, 7)]
    },
    'Expression': {
        'attr_idx': 16,
        'w_idxs': [(0, 17)]
    },
}

def replace_w_space(attr_name, current, target):
    w_idxs = ATTR_ID[attr_name]['w_idxs']
    new_w = current.clone().detach()
    for a, b in w_idxs:
        new_w[0][a:b + 1] = target[0][a:b + 1]
    return new_w

def set_normal(sf_model, save_dir, fn, edit_items):
    basename = fn[:fn.rfind('.')]
    cur_latent = torch.load(os.path.join('inversion/results', f'{basename}.pt'))
    cur_light = np.load(os.path.join('lights/results', f'{basename}.npy'))
    attr = json.load(open(os.path.join('attributes', 'attr.json')))
    cur_attr = np.array([
        [attr['Gender']],
        [attr['Glasses']],
        [attr['Yaw']],
        [attr['Pitch']],
        [attr['Baldness']],
        [attr['Beard']],
        [attr['Age']],
        [attr['Expression']],
    ])

    # initialize
    img_in, w_in, attr_in = sf_model.set_latents(cur_latent, cur_light, cur_attr)
    img_out, w_new, attr_new = img_in, w_in, attr_in

    if 'delight' in edit_items:
        target_light = np.array([[
            [[1.]],
            [[0.]],
            [[0.]],
            [[0.]],
            [[0.]],
            [[0.]],
            [[0.]],
            [[0.]],
            [[0.]],
        ]])
        # change lighting
        img_out, w_new, attr_new = sf_model.change_light(target_light)

    delight_result_img = Image.fromarray(img_out, 'RGB')
    delight_result_img.save(os.path.join('lights/results', f'{basename}_delight.png'))

    if 'norm_attr' in edit_items:
        target_attr = {
            'Glasses': np.array([0]),  # no glass
            'Yaw': np.array([0]),  # 0 degree
            'Pitch': np.array([0]),  # 0 degree
            'Baldness': np.array([1]),  # no hair
        }
        # change attributes
        for attr_name in target_attr.keys():
            img_out, w_new, attr_new = sf_model.change_attr(attr_name=attr_name, attr_value=target_attr[attr_name])

    if 'multi_yaw' in edit_items:
        # turn left, and given right face
        yaw = np.array([-30])
        img_out, w_new, attr_new = sf_model.change_attr(attr_name='Yaw', attr_value=yaw, keep_change=False)
        Image.fromarray(img_out, 'RGB').save(os.path.join(save_dir, f'{basename}_right.png'))
        torch.save({
            'latent': w_new,
            'attribute': attr_new
        }, os.path.join(save_dir, f'{basename}_right_latent.pt'))
        # turn right, and given left face
        yaw = np.array([30])
        img_out, w_new, attr_new = sf_model.change_attr(attr_name='Yaw', attr_value=yaw, keep_change=False)
        Image.fromarray(img_out, 'RGB').save(os.path.join(save_dir, f'{basename}_left.png'))
        torch.save({
            'latent': w_new,
            'attribute': attr_new
        }, os.path.join(save_dir, f'{basename}_left_latent.pt'))

    if 'multi_pitch' in edit_items:
        # turn down, and given up face
        pitch = np.array([-30])
        img_out, w_new, attr_new = sf_model.change_attr(attr_name='Pitch', attr_value=pitch, keep_change=False)
        Image.fromarray(img_out, 'RGB').save(os.path.join(save_dir, f'{basename}_up.png'))
        torch.save({
            'latent': w_new,
            'attribute': attr_new
        }, os.path.join(save_dir, f'{basename}_up_latent.pt'))
        # turn up, and given down face
        pitch = np.array([30])
        img_out, w_new, attr_new = sf_model.change_attr(attr_name='Pitch', attr_value=pitch, keep_change=False)
        Image.fromarray(img_out, 'RGB').save(os.path.join(save_dir, f'{basename}_down.png'))
        torch.save({
            'latent': w_new,
            'attribute': attr_new
        }, os.path.join(save_dir, f'{basename}_down_latent.pt'))

def editing(args):
    result_dir = 'editing/results'
    os.makedirs(result_dir, exist_ok=True)
    det_attributes(args)
    sf_model = StyleFlow(args)
    for fn in args.input_fnames:
        tic = time.time()
        set_normal(sf_model, result_dir, fn, args.edit_items.split(','))
        toc = time.time()
        print('Editing {} took {:.4f} seconds.'.format(fn, toc - tic))

class Build_model:

    def __init__(self, opt):

        self.opt = opt
        network_pkl = self.opt.network_pkl
        print('Loading networks from "%s"...' % network_pkl)
        _G, _D, Gs = load_networks(network_pkl)
        self.Gs = Gs
        self.Gs_syn_kwargs = dnnlib.EasyDict()
        self.Gs_syn_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
        self.Gs_syn_kwargs.randomize_noise = False
        self.Gs_syn_kwargs.minibatch_size = 4
        self.noise_vars = [var for name, var in Gs.components.synthesis.vars.items() if name.startswith('noise')]
        rnd = np.random.RandomState(0)
        tflib.set_vars({var: rnd.randn(*var.shape.as_list()) for var in self.noise_vars})

    def generate_im_from_random_seed(self, seed=22, truncation_psi=0.5):
        Gs = self.Gs
        seeds = [seed]
        noise_vars = [var for name, var in Gs.components.synthesis.vars.items() if name.startswith('noise')]

        Gs_kwargs = dnnlib.EasyDict()
        Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
        Gs_kwargs.randomize_noise = False
        if truncation_psi is not None:
            Gs_kwargs.truncation_psi = truncation_psi

        for seed_idx, seed in enumerate(seeds):
            print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
            rnd = np.random.RandomState(seed)
            z = rnd.randn(1, *Gs.input_shape[1:])  # [minibatch, component]
            tflib.set_vars({var: rnd.randn(*var.shape.as_list()) for var in noise_vars})  # [height, width]
            images = Gs.run(z, None, **Gs_kwargs)  # [minibatch, height, width, channel]
            # PIL.Image.fromarray(images[0], 'RGB').save(dnnlib.make_run_dir_path('seed%04d.png' % seed))
        return images

    def generate_im_from_z_space(self, z, truncation_psi=0.5):
        Gs = self.Gs

        Gs_kwargs = dnnlib.EasyDict()
        Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
        Gs_kwargs.randomize_noise = False
        if truncation_psi is not None:
            Gs_kwargs.truncation_psi = truncation_psi  # [height, width]

        images = Gs.run(z, None, **Gs_kwargs)
        # PIL.Image.fromarray(images[0], 'RGB').save(dnnlib.make_run_dir_path('test_from_z.png'))
        return images

    def generate_im_from_w_space(self, w):

        images = self.Gs.components.synthesis.run(w, **self.Gs_syn_kwargs)
        # PIL.Image.fromarray(images[0], 'RGB').save(dnnlib.make_run_dir_path('test_from_w.png'))
        return images

class StyleFlow:

    def __init__(self, args):

        super().__init__()
        self.device = args.device

        # StyleGAN model
        self.stylegan_model = Build_model(args)

        # StyleFlow model
        self.prior = cnf(512, '512-512-512-512-512', 17, 1)
        self.prior.load_state_dict(torch.load(args.flow_model_path))
        self.prior.eval()

        # Expression direction
        # self.exp_direct = torch.load(args.exp_direct_path).to(self.device)
        # self.exp_direct = self.exp_direct.unsqueeze(1).repeat([1, 18, 1])

        # Expression recognition model
        #self.exp_recon_model = Exp_Recog_API(model_path=args.exp_recognition_path)

        self.zero_padding = torch.zeros(1, 18, 1).to(self.device)
        self.z_current = None  # tensor [1, 18, 512]
        self.w_current = None  # tensor [1, 18, 512]
        self.attr_current = None  # tensor [1, 17, 1, 1]
        self.GAN_image = None  # array [1024, 1024, 3]

    def set_latents(self, curr_w, cur_light, cur_attr):
        '''
        curr_w: tensor [1, 18, 512]
        cur_light: array [1, 9, 1, 1]
        cur_attr: array [8, 1]
        '''
        self.w_current = curr_w.to(self.device)  # [1, 18, 512]
        self.attr_current = torch.cat(
            [
                torch.from_numpy(cur_light).type(torch.FloatTensor),
                torch.from_numpy(cur_attr).type(torch.FloatTensor).unsqueeze(0).unsqueeze(-1),
            ],
            dim=1,
        ).to(self.device)  # [1, 17, 1, 1]
        
        # get z from w
        self.z_current = self.prior(self.w_current, self.attr_current, logpx=self.zero_padding)[0]  # [1, 18, 512]
        self.GAN_image = self.stylegan_model.generate_im_from_w_space(
            self.w_current.detach().cpu().numpy())[0]  # [1024, 1024, 3] array

        return self.GAN_image, self.w_current, self.attr_current

    def change_light(self, lightingvec, keep_change=True):
        '''
        lightingvec: array [1, 9, 1, 1]
        keep_change: bool, if True, keep the changes to current state
        '''

        # replace attributes, :9 dim is light, and replace them by target light
        attr_target = self.attr_current.clone().detach().to(self.device)
        attr_target[:, :9] = torch.from_numpy(lightingvec).type(torch.FloatTensor).to(self.device)

        # get w_target
        w_target = self.prior(self.z_current, attr_target, logpx=self.zero_padding, reverse=True)[0]

        # replace w, in w+ space, only replace 7-11 layers
        w_target = replace_w_space(attr_name='Light', current=self.w_current, target=w_target)
        GAN_image = self.stylegan_model.generate_im_from_w_space(w_target.detach().cpu().numpy())[0]

        if keep_change:
            self.w_current = w_target.clone().detach().to(self.device)
            self.attr_current = attr_target.clone().detach().to(self.device)
            self.z_current = self.prior(self.w_current, self.attr_current, logpx=self.zero_padding)[0]
            self.GAN_image = GAN_image

        return GAN_image, w_target, attr_target

    def change_attr(self, attr_name, attr_value, keep_change=True):
        '''
        attr_name: str
        attr_value: array [1]
        keep_change: bool, if True, keep the changes to current state
        '''

        # replace attributes
        attr_target = self.attr_current.clone().detach().to(self.device)
        attr_idx = ATTR_ID[attr_name]['attr_idx']
        attr_target[:, attr_idx, 0, 0] = torch.from_numpy(attr_value).type(torch.FloatTensor).to(self.device)

        # get w_target
        w_target = self.prior(self.z_current, attr_target, logpx=self.zero_padding, reverse=True)[0]

        # replace w, in w+ space, only replace 7-11 layers
        w_target = replace_w_space(attr_name=attr_name, current=self.w_current, target=w_target)
        GAN_image = self.stylegan_model.generate_im_from_w_space(w_target.detach().cpu().numpy())[0]

        if keep_change:
            self.w_current = w_target.clone().detach().to(self.device)
            self.attr_current = attr_target.clone().detach().to(self.device)
            self.z_current = self.prior(self.w_current, self.attr_current, logpx=self.zero_padding)[0]
            self.GAN_image = GAN_image

        return GAN_image, w_target, attr_target