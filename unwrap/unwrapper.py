import os, dlib, skimage, cv2
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as tcl
from tensorflow.contrib.framework import arg_scope

def write_obj_with_texture(obj_name, vertices, triangles, uv_coords):
    ''' Save 3D face model with texture represented by texture map.
    Ref: https://github.com/patrikhuber/eos/blob/bd00155ebae4b1a13b08bf5a991694d682abbada/include/eos/core/Mesh.hpp
    Args:
        obj_name: str
        vertices: shape = (nver, 3)
        triangles: shape = (ntri, 3)
        texture: shape = (256,256,3)
        uv_coords: shape = (nver, 3) max value<=1
    '''

    if obj_name.split('.')[-1] != 'obj':
        obj_name = obj_name + '.obj'
    mtl_name = obj_name.replace('.obj', '.mtl')
    
    triangles = triangles.copy()
    triangles += 1 # mesh lab start with 1
    
    # write obj
    with open(obj_name, 'w') as f:
        # first line: write mtlib(material library)
        s = "mtllib {}\n".format(os.path.basename(mtl_name))
        f.write(s)

        # write vertices
        for i in range(vertices.shape[0]):
            s = 'v {} {} {}\n'.format(vertices[i, 0], vertices[i, 1], vertices[i, 2])
            f.write(s)
        
        # write uv coords
        for i in range(uv_coords.shape[0]):
            s = 'vt {} {}\n'.format(uv_coords[i,0], 1 - uv_coords[i,1])
            f.write(s)

        f.write("usemtl FaceTexture\n")

        # write f: ver ind/ uv ind
        for i in range(triangles.shape[0]):
            # s = 'f {}/{} {}/{} {}/{}\n'.format(triangles[i,0], triangles[i,0], triangles[i,1], triangles[i,1], triangles[i,2], triangles[i,2])
            s = 'f {}/{} {}/{} {}/{}\n'.format(triangles[i,2], triangles[i,2], triangles[i,1], triangles[i,1], triangles[i,0], triangles[i,0])
            f.write(s)

    # write mtl
    with open(mtl_name, 'w') as f:
        f.write("newmtl FaceTexture\n")
        s = 'map_Kd {}\n'.format('texture.png') # map to image
        f.write(s)

def resBlock(x, num_outputs, kernel_size = 4, stride=1, activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm, scope=None):
    assert num_outputs%2==0 #num_outputs must be divided by channel_factor(2 here)
    with tf.compat.v1.variable_scope(scope, 'resBlock'):
        shortcut = x
        if stride != 1 or x.get_shape()[3] != num_outputs:
            shortcut = tcl.conv2d(shortcut, num_outputs, kernel_size=1, stride=stride, 
                        activation_fn=None, normalizer_fn=None, scope='shortcut')
        x = tcl.conv2d(x, num_outputs/2, kernel_size=1, stride=1, padding='SAME')
        x = tcl.conv2d(x, num_outputs/2, kernel_size=kernel_size, stride=stride, padding='SAME')
        x = tcl.conv2d(x, num_outputs, kernel_size=1, stride=1, activation_fn=None, padding='SAME', normalizer_fn=None)

        x += shortcut       
        x = normalizer_fn(x)
        x = activation_fn(x)
    return x

class resfcn256(object):
    def __init__(self, resolution_inp = 256, resolution_op = 256, channel = 3, name = 'resfcn256'):
        self.name = name
        self.channel = channel
        self.resolution_inp = resolution_inp
        self.resolution_op = resolution_op

    def __call__(self, x, is_training = True):
        with tf.compat.v1.variable_scope(self.name) as scope:
            with arg_scope([tcl.batch_norm], is_training=is_training, scale=True):
                with arg_scope([tcl.conv2d, tcl.conv2d_transpose], activation_fn=tf.nn.relu, 
                                     normalizer_fn=tcl.batch_norm, 
                                     biases_initializer=None, 
                                     padding='SAME',
                                     weights_regularizer=tcl.l2_regularizer(0.0002)):
                    size = 16  
                    # x: s x s x 3
                    se = tcl.conv2d(x, num_outputs=size, kernel_size=4, stride=1) # 256 x 256 x 16
                    se = resBlock(se, num_outputs=size * 2, kernel_size=4, stride=2) # 128 x 128 x 32
                    se = resBlock(se, num_outputs=size * 2, kernel_size=4, stride=1) # 128 x 128 x 32
                    se = resBlock(se, num_outputs=size * 4, kernel_size=4, stride=2) # 64 x 64 x 64
                    se = resBlock(se, num_outputs=size * 4, kernel_size=4, stride=1) # 64 x 64 x 64
                    se = resBlock(se, num_outputs=size * 8, kernel_size=4, stride=2) # 32 x 32 x 128
                    se = resBlock(se, num_outputs=size * 8, kernel_size=4, stride=1) # 32 x 32 x 128
                    se = resBlock(se, num_outputs=size * 16, kernel_size=4, stride=2) # 16 x 16 x 256
                    se = resBlock(se, num_outputs=size * 16, kernel_size=4, stride=1) # 16 x 16 x 256
                    se = resBlock(se, num_outputs=size * 32, kernel_size=4, stride=2) # 8 x 8 x 512
                    se = resBlock(se, num_outputs=size * 32, kernel_size=4, stride=1) # 8 x 8 x 512

                    pd = tcl.conv2d_transpose(se, size * 32, 4, stride=1) # 8 x 8 x 512 
                    pd = tcl.conv2d_transpose(pd, size * 16, 4, stride=2) # 16 x 16 x 256 
                    pd = tcl.conv2d_transpose(pd, size * 16, 4, stride=1) # 16 x 16 x 256 
                    pd = tcl.conv2d_transpose(pd, size * 16, 4, stride=1) # 16 x 16 x 256 
                    pd = tcl.conv2d_transpose(pd, size * 8, 4, stride=2) # 32 x 32 x 128 
                    pd = tcl.conv2d_transpose(pd, size * 8, 4, stride=1) # 32 x 32 x 128 
                    pd = tcl.conv2d_transpose(pd, size * 8, 4, stride=1) # 32 x 32 x 128 
                    pd = tcl.conv2d_transpose(pd, size * 4, 4, stride=2) # 64 x 64 x 64 
                    pd = tcl.conv2d_transpose(pd, size * 4, 4, stride=1) # 64 x 64 x 64 
                    pd = tcl.conv2d_transpose(pd, size * 4, 4, stride=1) # 64 x 64 x 64 
                    
                    pd = tcl.conv2d_transpose(pd, size * 2, 4, stride=2) # 128 x 128 x 32
                    pd = tcl.conv2d_transpose(pd, size * 2, 4, stride=1) # 128 x 128 x 32
                    pd = tcl.conv2d_transpose(pd, size, 4, stride=2) # 256 x 256 x 16
                    pd = tcl.conv2d_transpose(pd, size, 4, stride=1) # 256 x 256 x 16

                    pd = tcl.conv2d_transpose(pd, 3, 4, stride=1) # 256 x 256 x 3
                    pd = tcl.conv2d_transpose(pd, 3, 4, stride=1) # 256 x 256 x 3
                    pos = tcl.conv2d_transpose(pd, 3, 4, stride=1, activation_fn = tf.nn.sigmoid)#, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))
                                
                    return pos
    @property
    def vars(self):
        #return [var for var in tf.global_variables() if self.name in var.name]
        return [var for var in tf.compat.v1.global_variables() if self.name in var.name]

class PosPrediction():
    def __init__(self, resolution_inp = 256, resolution_op = 256): 
        # -- hyper settings
        self.resolution_inp = resolution_inp
        self.resolution_op = resolution_op
        self.MaxPos = resolution_inp*1.1

        # network type
        self.network = resfcn256(self.resolution_inp, self.resolution_op)

        # net forward
        self.x = tf.compat.v1.placeholder(tf.float32, shape=[None, self.resolution_inp, self.resolution_inp, 3])
        
        self.x_op = self.network(self.x, is_training = False)
        self.sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True)))
        
    def restore(self, model_path):        
        tf.compat.v1.train.Saver(self.network.vars).restore(self.sess, model_path)
 
    def predict(self, image):
        pos = self.sess.run(self.x_op, 
                    feed_dict = {self.x: image[np.newaxis, :,:,:]})
        pos = np.squeeze(pos)
        return pos*self.MaxPos

    def predict_batch(self, images):
        pos = self.sess.run(self.x_op, 
                    feed_dict = {self.x: images})
        return pos*self.MaxPos

def unwrapping(args, resolution):
    result_dir = 'unwrap/results'
    os.makedirs(result_dir, exist_ok=True)
    
    pos_predictor = PosPrediction(resolution, resolution)
    pos_predictor.restore(args.prnet_model_path)
    
    input_img_paths = []
    for fn in args.input_fnames:
        input_img_paths.append(os.path.join(args.input_dir, fn))
        
        fn_left = fn.replace('.png', '_left.png')
        input_img_paths.append(os.path.join('editing', 'results', fn_left))

        fn_right = fn.replace('.png', '_right.png')
        input_img_paths.append(os.path.join('editing', 'results', fn_right))
        
    for path in input_img_paths:
        img = dlib.load_rgb_image(path)
        [h, w, _] = img.shape
        
        # preprocessing (alignment)
        max_size = max(h, w)
        if max_size > 1000:
            img = skimage.transform.rescale(img, 1000./max_size, channel_axis=2)
            img = (img*255).astype(np.uint8)
        detector = dlib.get_frontal_face_detector()
        detected_faces = detector(img, 1)
        if len(detected_faces) == 0:
            print('warning: no detected face')
        #img = img/255.
        d = detected_faces[0] ## only use the first detected face (assume that each input image only contains one face)
        left = d.left(); right = d.right(); top = d.top(); bottom = d.bottom()
        old_size = (right - left + bottom - top)/2
        center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0 + old_size*0.14])
        size = int(old_size*1.58)
        src_pts = np.array([[center[0]-size/2, center[1]-size/2], [center[0]-size/2, center[1]+size/2], [center[0]+size/2, center[1]-size/2]])
        DST_PTS = np.array([[0,0], [0,resolution - 1], [resolution - 1, 0]])
        tform = skimage.transform.estimate_transform('similarity', src_pts, DST_PTS)
        cropped_image = skimage.transform.warp(img, tform.inverse, output_shape=(resolution, resolution))
        
        ############# POSITION MAP REGRESSION #############
        cropped_pos = pos_predictor.predict(cropped_image)
        ##################################################
        
        # interpolation
        cropped_vertices = np.reshape(cropped_pos, [-1, 3]).T
        z = cropped_vertices[2,:].copy()/tform.params[0,0]
        cropped_vertices[2,:] = 1
        vertices = np.dot(np.linalg.inv(tform.params), cropped_vertices)
        vertices = np.vstack((vertices[:2,:], z))
        pos = np.reshape(vertices.T, [resolution, resolution, 3])
        pos_interpolated = pos.copy()
        texture = cv2.remap(img, pos_interpolated[:,:,:2].astype(np.float32), None, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0))
        
        fn = os.path.basename(path)
        fn_tex = fn.replace('.png', '_tex.png')
        dlib.save_image(texture.astype(np.uint8), os.path.join(result_dir, fn_tex))

        # create mesh object
        if fn.find('_left') == -1 and fn.find('_right') == -1:
            prnet_model_dir = os.path.dirname(args.prnet_model_path)
            face_ind = np.loadtxt(os.path.join(prnet_model_dir,'face_ind.txt')).astype(np.int32) # get valid vertices in the pos map
            all_vertices = np.reshape(pos_interpolated, [resolution**2, -1])
            vertices = all_vertices[face_ind, :]
            vertices[:,1] = h - 1 - vertices[:,1]

            triangles = np.loadtxt(os.path.join(prnet_model_dir,'triangles.txt')).astype(np.int32)

            uvcoords = np.meshgrid(range(resolution),range(resolution))
            uvcoords = np.transpose(np.array(uvcoords), [1,2,0])
            uvcoords = np.reshape(uvcoords, [resolution**2, -1])
            uvcoords = uvcoords[face_ind, :]
            uvcoords = np.hstack((uvcoords[:,:2], np.zeros([uvcoords.shape[0], 1])))

            dirname = os.path.splitext(fn)[0]
            output_dir = os.path.join(args.output_dir, dirname)
            os.makedirs(output_dir, exist_ok=True)
            np.save(os.path.join(output_dir, 'vertices.npy'), vertices)
            np.save(os.path.join(output_dir, 'triangles.npy'), triangles)
            np.save(os.path.join(output_dir, 'uvcoords.npy'), uvcoords)
            write_obj_with_texture(os.path.join(output_dir, 'mesh.obj'), vertices, triangles, uvcoords/resolution)