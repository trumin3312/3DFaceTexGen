import argparse

class TestOptions():

    def initialize(self, parser):

        parser.set_defaults(phase='test', focal=1015.0, camera_distance=10.0)
        parser.add_argument("--device", type=str, default='cuda:0', help="The device, optional: cpu/cuda.")
        parser.add_argument("--input_dir", type=str, default='inputs', help="The directory of the attributes.")
        parser.add_argument("--output_dir", type=str, default='outputs', help="The name of the saved folder.")
        
        ### Step1_Inversion ###
        parser.add_argument("--e4e_model_path",
                            type=str,
                            default='checkpoints/e4e_model/e4e_ffhq_encode.pt',
                            help="The path of the pretrained e4e model.")

        parser.add_argument("--shape_predictor_model_path",
                            type=str,
                            default='checkpoints/dlib_model/shape_predictor_68_face_landmarks.dat',
                            help="The path of the dlib shape predictor model (68 face landmarks).")
        
        ### Step2_Editing ###
        parser.add_argument("--dpr_model_path",
                            type=str,
                            default='checkpoints/dpr_model/trained_model_03.t7',
                            help="The path of the pretrained DPR model.")
        
        parser.add_argument("--flow_model_path",
                            type=str,
                            default='checkpoints/styleflow_model/modellarge10k.pt',
                            help="The path of the pretrained styleflow model.")

        parser.add_argument('--network_pkl',
                            type=str,
                            default='checkpoints/stylegan_model/stylegan2-ffhq-config-f.pkl')

        parser.add_argument("--edit_items",
                            type=str,
                            default='delight,norm_attr,multi_yaw',
                            help="The edited items, optional:[delight, norm_attr, multi_yaw, multi_pitch], joint by ','.")
        
        parser.add_argument("--prnet_model_path",
                            type=str,
                            default='checkpoints/prnet_model/256_256_resfcn256_weight',
                            help="The path of the pretrained e4e model.")
        
        return parser

    def gather_options(self):
        # initialize parser with basic options
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser = self.initialize(parser)
        opt = parser.parse_args('')
        self.parser = parser
        return opt

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)
    
    def parse(self):

        opt = self.gather_options()

        self.print_options(opt)

        self.opt = opt
        return self.opt