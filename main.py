import argparse
import os
from pathlib import Path

import cv2
import dlib
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.functional as nnf
from model.bisenet.model import BiSeNet
from model.encoder.align_all_parallel import align_face
from model.vtoonify import VToonify
from torchvision import transforms
from util import get_video_crop_parameter
from util import load_psp_standalone
from util import save_image
from ts.torch_handler.base_handler import BaseHandler
from ts.context import Context

SLIDING_WINDOW_SIZE = 2


class Arguments():
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Style Transfer')
        self.parser.add_argument('--content', type=str, default='./data', help='path of the folder with the content image/video')
        self.parser.add_argument('--style_id', type=int, default=26, help='the id of the style image')
        self.parser.add_argument('--style_degree', type=float, default=0.5, help='style degree for VToonify-D')
        self.parser.add_argument('--color_transfer', action='store_true', help='transfer the color of the style')
        self.parser.add_argument('--ckpt', type=str, default='./checkpoint/vtoonify_d_cartoon/vtoonify_s_d.pt', help='path of the saved model')
        self.parser.add_argument('--output_path', type=str, default='./output/', help='path of the output images')
        self.parser.add_argument('--scale_image', action='store_true', help='resize and crop the image to best fit the model')
        self.parser.add_argument('--style_encoder_path', type=str, default='./checkpoint/encoder.pt', help='path of the style encoder')
        self.parser.add_argument('--exstyle_path', type=str, default=None, help='path of the extrinsic style code')
        self.parser.add_argument('--faceparsing_path', type=str, default='./checkpoint/faceparsing.pth', help='path of the face parsing model')
        self.parser.add_argument('--cpu', action='store_true', help='if true, only use cpu')
        self.parser.add_argument('--backbone', type=str, default='dualstylegan', help='dualstylegan | toonify')
        self.parser.add_argument('--padding', type=int, nargs=4, default=[200, 200, 200, 200], help='left, right, top, bottom paddings to the face center')
        self.parser.add_argument('--skip_vtoonify', action='store_true', help='Skip VToonify Styling and create final image only with generator model')
        self.parser.add_argument('--psp_style', type=int, nargs='*', help='Mix face and style after pSp encoding', default=[])

    def parse(self):
        self.opt = self.parser.parse_args()
        if self.opt.exstyle_path is None:
            self.opt.exstyle_path = os.path.join(os.path.dirname(self.opt.ckpt), 'exstyle_code.npy')
        return self.opt

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
        self.backbone = self.manifest['models']['backbone']
        self.vtoonify = VToonify(backbone=self.backbone)
        self.vtoonify.load_state_dict(torch.load(self.manifest['models']['vtoonify'], map_location=lambda storage, loc: storage)['g_ema'])
        self.vtoonify.to(self.device)

        self.parsingpredictor = BiSeNet(n_classes=19)
        self.parsingpredictor.load_state_dict(torch.load(self.manifest['models']['faceparsing'], map_location=lambda storage, loc: storage))
        self.parsingpredictor.to(self.device).eval()

        # Load Landmark Predictor model
        face_landmark_modelname = './checkpoint/shape_predictor_68_face_landmarks.dat'
        if not os.path.exists(face_landmark_modelname):
            import wget
            import bz2
            wget.download('http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2', face_landmark_modelname+'.bz2')
            zipfile = bz2.BZ2File(face_landmark_modelname+'.bz2')
            data = zipfile.read()
            open(face_landmark_modelname, 'wb').write(data)
        self.landmarkpredictor = dlib.shape_predictor(face_landmark_modelname)

        # Load pSp
        self.pspencoder = load_psp_standalone(self.manifest['models']['style_encoder'], self.device)

        # Load External Styles
        if self.backbone == 'dualstylegan':
            self.exstyles = np.load(self.manifest['models']['exstyle'], allow_pickle='TRUE').item()
            stylename = list(self.exstyles.keys())[self.manifest['models']['style_id']]
            self.exstyle = torch.tensor(self.exstyles[stylename]).to(self.device)
            with torch.no_grad():
                self.exstyle = self.vtoonify.zplus2wplus(self.exstyle)


        self.initialized = True

    def handle(self, filename, scale_image, padding, latent_mask, style_degree, skip_vtoonify):
        self.latent_mask = latent_mask
        self.style_degree = style_degree
        self.skip_vtoonify = skip_vtoonify

        # Load image
        frame = cv2.imread(filename)

        # Preprocess Image
        with torch.no_grad():
            frame = self.pre_processingImage(frame, scale_image, padding)

            model_output = self.inference(frame)

        return model_output

    def pre_processingImage(self, frame, scale_image, padding):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # We detect the face in the image, and resize the image so that the eye distance is 64 pixels.
        # Centered on the eyes, we crop the image to almost 400x400 (based on args.padding).
        if scale_image:
            paras = get_video_crop_parameter(frame, self.landmarkpredictor, padding)
            if paras is not None:
                h, w, top, bottom, left, right, scale = paras
                # for HR image, we apply gaussian blur to it to avoid over-sharp stylization results
                if scale <= 0.75:
                    frame = cv2.sepFilter2D(frame, -1, kernel_1d, kernel_1d)
                if scale <= 0.375:
                    frame = cv2.sepFilter2D(frame, -1, kernel_1d, kernel_1d)
                frame = cv2.resize(frame, (w, h))[top:bottom, left:right]

        frame = align_face(frame, self.landmarkpredictor)
        frame = self.transform(frame).unsqueeze(dim=0).to(self.device)

        return frame

    def inference(self, model_input):
        # Encode Image
        s_w = self.encode_face_img(model_input)

        # Stylize pSp image
        print('Stylizing image with pSp')
        s_w = self.applyExstyle(s_w, self.exstyle, self.latent_mask)

        self.embeddings_buffer.append(torch.squeeze(s_w))
        self.window_slide()

        # Compute Mean
        mean_s_w = self.pSpFeaturesBufferMean()

        # Update VToonify Frame to mean face
        print('Decoding mean image')
        original_frame_size = model_input.shape[:2]
        frame = self.decodeFeaturesToImg(mean_s_w)

        if self.skip_vtoonify:
            return None, frame

        print('Using VToonify to stylize image')
        # Resize frame to save memory
        frame = cv2.resize(frame, original_frame_size)

        vtoonfy_output_image = self.apply_vtoonify(frame, mean_s_w)
        return vtoonfy_output_image, frame

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
        if len(self.embeddings_buffer) > SLIDING_WINDOW_SIZE:
            self.embeddings_buffer.pop(0)

    def pSpFeaturesBufferMean(self):
        if len(self.embeddings_buffer) > 1:
            all_latents = torch.stack(self.embeddings_buffer, dim=0)
            s_w = torch.mean(all_latents, dim=0)
        else:
            s_w = self.embeddings_buffer[0]
        s_w = s_w.unsqueeze(0)

        return s_w

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
            if args.color_transfer:
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

if __name__ == '__main__':
    parser = Arguments()
    args = parser.parse()
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
        }
    }
    context = Context(model_dir=model_dir, model_name="vtoonify", manifest=manifest,batch_size=1,gpu=0,mms_version="1.0.0")

    vtoonify_handler = VToonifyHandler()
    vtoonify_handler.initialize(context)

    print('Loaded models successfully!')

    # Constants and variables
    scale = 1
    kernel_1d = np.array([[0.125], [0.375], [0.375], [0.125]])

    Path(args.output_path).mkdir(parents=True, exist_ok=True)  # Creates the output folder in case it does not exists
    for file in Path(args.content).glob('*'):
        filename = args.content + '/' + file.name
        basename = os.path.basename(filename).split('.')[0]
        
        # Save paths
        cropname = os.path.join(args.output_path, basename + '_input.jpg')
        savename = os.path.join(args.output_path, basename + '_vtoonify_' + args.backbone[0] + '.jpg')
        sum_savename = os.path.join(args.output_path, basename + '_vtoonify_SUM_' + args.backbone[0] + '.jpg')

        print('Processing ' + os.path.basename(filename) + ' with vtoonify_' + args.backbone[0])
        cropimage, result_image = vtoonify_handler.handle(
            filename,
            args.scale_image,
            args.padding,
            args.psp_style,
            args.style_degree,
            args.skip_vtoonify,
        )

        output_path = sum_savename if args.skip_vtoonify else savename
        result_image = cv2.imwrite(output_path, cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
        if cropimage is not None:
            cv2.imwrite(cropname, cv2.cvtColor(cropimage, cv2.COLOR_RGB2BGR))
