import argparse
import os
from pathlib import Path

import cv2
import torch
import dlib
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
if 'MAR-INF' in os.listdir():
    print('Inside Trochserve')
    os.environ['USING_TORCHSERVE'] = 'true'
if 'USING_TORCHSERVE' in os.environ:
    from align_all_parallel import align_face
    from bisnet_model import BiSeNet
    from vtoonify import VToonify
else:
    from model.bisenet.bisnet_model import BiSeNet
    from model.encoder.align_all_parallel import align_face
    from model.vtoonify import VToonify
from torchvision import transforms
from util import get_video_crop_parameter
from util import load_psp_standalone
from ts.torch_handler.base_handler import BaseHandler
from ts.context import Context
from util import interpolate
import base64


def setup_parser():
    parser = argparse.ArgumentParser(description='Style Transfer')
    
    # Add arguments to the parser
    parser.add_argument('--content', type=str, default='./data', help='path of the folder with the content image/video')
    # ... (add the rest of the arguments in a similar way)


    parser = argparse.ArgumentParser(description='Style Transfer')
    parser.add_argument('--content', type=str, default='./data', help='path of the folder with the content image/video')
    parser.add_argument('--style_id', type=int, default=26, help='the id of the style image')
    parser.add_argument('--style_degree', type=float, default=0.5, help='style degree for VToonify-D')
    parser.add_argument('--color_transfer', action='store_true', help='transfer the color of the style')
    parser.add_argument('--ckpt', type=str, default='./checkpoint/vtoonify_d_cartoon/vtoonify_s_d.pt', help='path of the saved model')
    parser.add_argument('--output_path', type=str, default='./output/', help='path of the output images')
    parser.add_argument('--scale_image', action='store_true', help='resize and crop the image to best fit the model')
    parser.add_argument('--style_encoder_path', type=str, default='./checkpoint/encoder.pt', help='path of the style encoder')
    parser.add_argument('--exstyle_path', type=str, default=None, help='path of the extrinsic style code')
    parser.add_argument('--faceparsing_path', type=str, default='./checkpoint/faceparsing.pth', help='path of the face parsing model')
    parser.add_argument('--cpu', action='store_true', help='if true, only use cpu')
    parser.add_argument('--backbone', type=str, default='dualstylegan', help='dualstylegan | toonify')
    parser.add_argument('--padding', type=int, nargs=4, default=[200, 200, 200, 200], help='left, right, top, bottom paddings to the face center')
    parser.add_argument('--skip_vtoonify', action='store_true', help='Skip VToonify Styling and create final image only with generator model')
    parser.add_argument('--psp_style', type=int, nargs='*', help='Mix face and style after pSp encoding', default=[])

    return parser

def parse(parser):
    opt = parser.parse_args()
    if opt.exstyle_path is None:
        opt.exstyle_path = os.path.join(os.path.dirname(opt.ckpt), 'exstyle_code.npy')
    return opt

class VToonifyHandler(BaseHandler): # for TorchServe  it need to inherit from BaseHandler
    """
    A custom model handler implementation.
    """

    def __init__(self):
        self._context = None
        self.initialized = False
        self.explain = False
        self.target = 0
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        self.embeddings_buffer = []
        self.vtoonify_input_image_size = (256,256)
        self.FPS = 25
        self.SLIDING_WINDOW_SIZE = 2
        self.duration_per_image = 1
        self.scale_image = True
        self.padding = [200, 200, 200, 200]
        self.kernel_1d = np.array([[0.125], [0.375], [0.375], [0.125]])

    def initialize(self, context):
        """
        Initialize model. This will be called during model loading time
        :param context: Initial context contains model server system properties.
        :return:
        """

        # From Torchserve on how to init. I guess manifest and other variables are necessary for propper working.
        # Content will be handled by Torchserve and include the necessary information
        self._context = context

        #  load the model
        self.manifest = context.manifest

        properties = context.system_properties
        model_dir = properties.get("model_dir")
        self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")
        
        # Read model serialize/pt file
        # serialized_file = self.manifest['model']['serializedFile']
        # model_pt_path = os.path.join(model_dir, serialized_file)

        # Load VToonify
        if 'USING_TORCHSERVE' in os.environ: # if we are realy in torchserve
            self.backbone = "dualstylegan"
            vtoonify_path = "vtoonify_s_d.pt"
            faceparsing_path = "faceparsing.pth"
            face_landmark_modelname = 'shape_predictor_68_face_landmarks.dat'
            style_encoder_path = "encoder.pt"
            exstyle_path = "exstyle_code.npy"
            self.latent_mask = [10,11,12,13,14]
            self.style_degree = 0.0
            self.skip_vtoonify = True
        else:
            self.backbone = self.manifest['models']['backbone']
            vtoonify_path = self.manifest['models']['vtoonify']
            faceparsing_path = self.manifest['models']['faceparsing']
            face_landmark_modelname = './checkpoint/shape_predictor_68_face_landmarks.dat'
            style_encoder_path = self.manifest['models']['style_encoder']
            exstyle_path = self.manifest['models']['exstyle']
            self.latent_mask = self.manifest['latent_mask']
            self.style_degree = self.manifest['style_degree']
            self.skip_vtoonify = self.manifest['skip_vtoonify']

        self.vtoonify = VToonify(backbone=self.backbone)
        self.vtoonify.load_state_dict(torch.load(vtoonify_path, map_location=lambda storage, loc: storage)['g_ema'])
        self.vtoonify.to(self.device)
        self.color_transfer = True

        self.parsingpredictor = BiSeNet(n_classes=19)
        self.parsingpredictor.load_state_dict(torch.load(faceparsing_path, map_location=lambda storage, loc: storage))
        self.parsingpredictor.to(self.device).eval()

        # Load Landmark Predictor model
        if not os.path.exists(face_landmark_modelname):
            import wget
            import bz2
            wget.download('http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2', face_landmark_modelname+'.bz2')
            zipfile = bz2.BZ2File(face_landmark_modelname+'.bz2')
            data = zipfile.read()
            open(face_landmark_modelname, 'wb').write(data)
        self.landmarkpredictor = dlib.shape_predictor(face_landmark_modelname)

        # Load pSp
        self.pspencoder = load_psp_standalone(style_encoder_path, self.device)

        # Load External Styles
        if self.backbone == 'dualstylegan':
            self.style_id = -1
            self.exstyles = np.load(exstyle_path, allow_pickle='TRUE').item()
            stylename = list(self.exstyles.keys())[self.style_id]
            self.exstyle = torch.tensor(self.exstyles[stylename]).to(self.device)
            with torch.no_grad():
                self.exstyle = self.vtoonify.zplus2wplus(self.exstyle)


        self.initialized = True

    def handle(self, data, context):
        # Load image
        image_bytes = data[0]['body']
        image_array = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        # Preprocess Image
        with torch.no_grad():
            model_input = self.pre_processingImage(frame)

            # Encode Image
            s_w = self.encode_face_img(model_input)

            # Stylize pSp image
            print('Stylizing image with pSp')
            s_w = self.applyExstyle(s_w, self.exstyle, self.latent_mask)

            self.embeddings_buffer.append(torch.squeeze(s_w))
            self.window_slide()

            mean_s_w = self.pSpFeaturesBufferMean()

            animation_frames = self.generate_animation([self.embeddings_buffer[-1], mean_s_w])
            model_output = animation_frames

        return self.postprocess(model_output)

    def pre_processingImage(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # We detect the face in the image, and resize the image so that the eye distance is 64 pixels.
        # Centered on the eyes, we crop the image to almost 400x400 (based on args.padding).
        if self.scale_image:
            paras = get_video_crop_parameter(frame, self.landmarkpredictor, self.padding)
            if paras is not None:
                h, w, top, bottom, left, right, scale = paras
                # for HR image, we apply gaussian blur to it to avoid over-sharp stylization results
                if scale <= 0.75:
                    frame = cv2.sepFilter2D(frame, -1, self.kernel_1d, self.kernel_1d)
                if scale <= 0.375:
                    frame = cv2.sepFilter2D(frame, -1, self.kernel_1d, self.kernel_1d)
                frame = cv2.resize(frame, (w, h))[top:bottom, left:right]

        frame = align_face(frame, self.landmarkpredictor)
        frame = self.transform(frame).unsqueeze(dim=0).to(self.device)

        return frame

    def s_w_to_stylized_image(self, s_w):
        # Update VToonify Frame to mean face
        print('Decoding mean image')
        frame = self.decodeFeaturesToImg(s_w)

        if self.skip_vtoonify:
            return frame

        print('Using VToonify to stylize image')
        # Resize frame to save memory
        frame = cv2.resize(frame, self.vtoonify_input_image_size)

        vtoonfy_output_image = self.apply_vtoonify(frame, s_w)
        return vtoonfy_output_image

    def encode_face_img(self, face_image):
        s_w = self.pspencoder(face_image)
        s_w = self.vtoonify.zplus2wplus(s_w)

        return s_w

    @staticmethod
    def applyExstyle(s_w, exstyle, latent_mask):
        if exstyle is None:
            print('No exstyle, skipping pSp styling')
            return s_w

        for i in latent_mask:
            s_w[:, i] = exstyle[:, i]

        return s_w

    def window_slide(self):
        if len(self.embeddings_buffer) > self.SLIDING_WINDOW_SIZE:
            self.embeddings_buffer.pop(0)

    def pSpFeaturesBufferMean(self):
        if len(self.embeddings_buffer) > 1:
            all_latents = torch.stack(self.embeddings_buffer, dim=0)
            s_w = torch.mean(all_latents, dim=0)
        else:
            s_w = self.embeddings_buffer[0]
        s_w = s_w.unsqueeze(0)

        return s_w

    def generate_animation(self, encodings, FPS=25, duration_per_image=1):
        encodings = [encoding.squeeze() for encoding in encodings]
        animation_frames = []

        for s_w in tqdm(interpolate(
                latents_list=encodings, duration_list=[duration_per_image]*len(encodings),
                interpolation_type="linear",
                loop=False,
                FPS=FPS,
            ), desc='Generating morphing animation'):
            animation_frame = self.s_w_to_stylized_image(s_w)
            animation_frames.append(animation_frame)

        return animation_frames

    def decodeFeaturesToImg(self, s_w):
        frame_tensor, _ = self.vtoonify.generator.generator([s_w], input_is_latent=True, randomize_noise=True)
        # frame_tensor, _ = self.vtoonify.generator([s_w], s_w, input_is_latent=True, randomize_noise=True, use_res=False)
        frame = ((frame_tensor[0].detach().cpu().numpy().transpose(1, 2, 0) + 1.0) * 127.5).astype(np.uint8)
        # frame = frame_tensor.detach().cpu().numpy().astype(np.uint8)
        return frame

    def apply_vtoonify(self, frame, s_w):
        # Compute VToonify Features
        s_w, inputs = self.processingStyle(frame, s_w)

        # Process Image with VToonify
        y_tilde = self.vtoonify(inputs, s_w.repeat(inputs.size(0), 1, 1), d_s=self.style_degree)
        y_tilde = torch.clamp(y_tilde, -1, 1)

        # Save Output Image
        output_image = self.normalize_image(y_tilde[0].cpu())

        return output_image

    def processingStyle(self, frame, s_w):
        s_w = self.vtoonify.zplus2wplus(s_w)
        if self.vtoonify.backbone == 'dualstylegan':
            if self.color_transfer:
                s_w = self.exstyle
            else:
                s_w[:, :7] = self.exstyle[:, :7]

        x = self.transform(frame).unsqueeze(dim=0).to(self.device)
        # parsing network works best on 512x512 images, so we predict parsing maps on upsmapled frames
        # followed by downsampling the parsing maps
        x_p = F.interpolate(
            self.parsingpredictor(2*(F.interpolate(
                x,
                scale_factor=2,
                mode='bilinear',
                align_corners=False)))[0],
            scale_factor=0.5,
            recompute_scale_factor=False).detach()
        # we give parsing maps lower weight (1/16)
        inputs = torch.cat((x, x_p/16.), dim=1)

        return s_w, inputs

    @staticmethod
    def normalize_image(img):
        tmp = ((img.detach().cpu().numpy().transpose(1, 2, 0) + 1.0) * 127.5).astype(np.uint8)
        return tmp

    def postprocess(self, inference_output):
        """
        Return inference result.
        :param inference_output: list of inference output
        :return: list of predict results
        """
        # Take output from network and post-process to desired format
        
        encoded_frames = [self.image_to_base64(image) for image in inference_output]
        postprocess_output = {'video_frames': encoded_frames}
        # postprocess_output = {'video_frames': [encoded_frames[0], encoded_frames[-1]]}
        return [postprocess_output]

    @staticmethod
    def image_to_base64(image):
        print('Encoding image')
        _, im_arr = cv2.imencode('.JPEG', image, [int(cv2.IMWRITE_JPEG_QUALITY), 80])  # im_arr: image in Numpy one-dim array format.
        byte_arr = im_arr.tobytes()
        return base64.b64encode(byte_arr).decode('utf-8')

if __name__ == '__main__':
    parser = setup_parser()
    args = parse(parser)
    print('Loaded arguments')
    for name, value in sorted(vars(args).items()):
        print('%s: %s' % (str(name), str(value)))
    print('*'*98)

    device = 'cpu' if args.cpu else 'cuda'
    model_dir = "pretrained_models"
    model_name = "psp_ffhq_encode.pt"
    manifest = {
        'models': {
            'vtoonify': args.ckpt,
            'faceparsing': args.faceparsing_path,
            'style_encoder': args.style_encoder_path,
            'backbone': args.backbone,
            'exstyle': args.exstyle_path,
            'style_id': args.style_id,
        },
        'latent_mask': args.psp_style,
        'style_degree': args.style_degree,
        'skip_vtoonify': args.skip_vtoonify,
    }
    context = Context(model_dir=model_dir, model_name="vtoonify", manifest=manifest,batch_size=1,gpu=0,mms_version="1.0.0")

    vtoonify_handler = VToonifyHandler()
    vtoonify_handler.initialize(context)

    print('Loaded models successfully!')

    Path(args.output_path).mkdir(parents=True, exist_ok=True)  # Creates the output folder in case it does not exists
    for file in Path(args.content).glob('*'):
        filename = args.content + '/' + file.name
        basename = os.path.basename(filename).split('.')[0]

        with open(filename, "rb") as image:
            f = image.read()
            input_image = bytearray(f)

        print('Processing ' + os.path.basename(filename) + ' with vtoonify_' + args.backbone[0])
        output_image = vtoonify_handler.handle(input_image, context)

        model_response = vtoonify_handler.handle([{'body': input_image}], context)

        test_output_path = args.output_path + '/' + basename
        os.makedirs(test_output_path, exist_ok=True)
        for i, frame_base64 in enumerate(model_response[0]['video_frames']):
            frame_bytes = base64.b64decode(frame_base64)
            frame_array = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
            
            frame_path = f'{test_output_path}/{i}.jpeg'
            cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)) # save only last frame for test
