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
        self.parser.add_argument('--psp_style', action='store_true', help='Mix face and style after pSp encoding')

    def parse(self):
        self.opt = self.parser.parse_args()
        if self.opt.exstyle_path is None:
            self.opt.exstyle_path = os.path.join(os.path.dirname(self.opt.ckpt), 'exstyle_code.npy')
        return self.opt


def window_slide():
    if len(embeddings_buffer) > SLIDING_WINDOW_SIZE:
        embeddings_buffer.pop(0)


def pre_processingImage(args, filename, basename, landmarkpredictor):
    cropname = os.path.join(args.output_path, basename + '_input.jpg')
    savename = os.path.join(args.output_path, basename + '_vtoonify_' + args.backbone[0] + '.jpg')
    sum_savename = os.path.join(args.output_path, basename + '_vtoonify_SUM_' + args.backbone[0] + '.jpg')

    frame = cv2.imread(filename)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # We detect the face in the image, and resize the image so that the eye distance is 64 pixels.
    # Centered on the eyes, we crop the image to almost 400x400 (based on args.padding).
    if args.scale_image:
        paras = get_video_crop_parameter(frame, landmarkpredictor, args.padding)
        if paras is not None:
            h, w, top, bottom, left, right, scale = paras
            # for HR image, we apply gaussian blur to it to avoid over-sharp stylization results
            if scale <= 0.75:
                frame = cv2.sepFilter2D(frame, -1, kernel_1d, kernel_1d)
            if scale <= 0.375:
                frame = cv2.sepFilter2D(frame, -1, kernel_1d, kernel_1d)
            frame = cv2.resize(frame, (w, h))[top:bottom, left:right]

    return cropname, savename, sum_savename, frame


def encode_face_img(device, frame, landmarkpredictor):
    I = align_face(frame, landmarkpredictor)
    I = transform(I).unsqueeze(dim=0).to(device)

    s_w = pspencoder(I)

    return s_w


def processingStyle(device, frame, s_w):
    s_w = vtoonify.zplus2wplus(s_w)
    if vtoonify.backbone == 'dualstylegan':
        if args.color_transfer:
            s_w = exstyle
        else:
            s_w[:, :7] = exstyle[:, :7]

    x = transform(frame).unsqueeze(dim=0).to(device)
    # parsing network works best on 512x512 images, so we predict parsing maps on upsmapled frames
    # followed by downsampling the parsing maps
    x_p = F.interpolate(
        parsingpredictor(2*(F.interpolate(
            x,
            scale_factor=2,
            mode='bilinear',
            align_corners=False)))[0],
        scale_factor=0.5,
        recompute_scale_factor=False).detach()
    # we give parsing maps lower weight (1/16)
    inputs = torch.cat((x, x_p/16.), dim=1)

    return s_w, inputs


def pSpFeaturesBufferMean(features_buffer):
    if len(features_buffer) > 1:
        all_latents = torch.stack(features_buffer, dim=0)
        s_w = torch.mean(all_latents, dim=0)
    else:
        s_w = features_buffer[0]
    return s_w.unsqueeze(0)


def decodeFeaturesToImg(s_w, vtoonify):
    s_w = vtoonify.zplus2wplus(s_w)
    frame_tensor, _ = vtoonify.generator.generator([s_w], input_is_latent=True, randomize_noise=True)
    # frame_tensor, _ = vtoonify.generator([s_w], s_w, input_is_latent=True, randomize_noise=True, use_res=False)
    frame = ((frame_tensor[0].detach().cpu().numpy().transpose(1, 2, 0) + 1.0) * 127.5).astype(np.uint8)
    # frame = frame_tensor.detach().cpu().numpy().astype(np.uint8)
    return frame


def applyExstyle(s_w, exstyle, latent_mask):
    if exstyle is None:
        print('No exstyle, skipping pSp styling')
        return s_w

    for i in latent_mask:
        s_w[:, i] = exstyle[:, i]

    return s_w


if __name__ == '__main__':
    parser = Arguments()
    args = parser.parse()
    print('Loaded arguments')
    for name, value in sorted(vars(args).items()):
        print('%s: %s' % (str(name), str(value)))
    print('*'*98)

    device = 'cpu' if args.cpu else 'cuda'

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    # Load VToonify
    vtoonify = VToonify(backbone=args.backbone)
    vtoonify.load_state_dict(torch.load(args.ckpt, map_location=lambda storage, loc: storage)['g_ema'])
    vtoonify.to(device)

    parsingpredictor = BiSeNet(n_classes=19)
    parsingpredictor.load_state_dict(torch.load(args.faceparsing_path, map_location=lambda storage, loc: storage))
    parsingpredictor.to(device).eval()

    # Load Landmark Predictor model
    face_landmark_modelname = './checkpoint/shape_predictor_68_face_landmarks.dat'
    if not os.path.exists(face_landmark_modelname):
        import wget
        import bz2
        wget.download('http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2', face_landmark_modelname+'.bz2')
        zipfile = bz2.BZ2File(face_landmark_modelname+'.bz2')
        data = zipfile.read()
        open(face_landmark_modelname, 'wb').write(data)
    landmarkpredictor = dlib.shape_predictor(face_landmark_modelname)

    # Load pSp
    pspencoder = load_psp_standalone(args.style_encoder_path, device)

    # Load External Styles
    if args.backbone == 'dualstylegan':
        exstyles = np.load(args.exstyle_path, allow_pickle='TRUE').item()
        stylename = list(exstyles.keys())[args.style_id]
        exstyle = torch.tensor(exstyles[stylename]).to(device)
        with torch.no_grad():
            exstyle = vtoonify.zplus2wplus(exstyle)

    print('Loaded models successfully!')

    # Constants and variables
    scale = 1
    kernel_1d = np.array([[0.125], [0.375], [0.375], [0.125]])
    latent_mask = [10,11,12,13,14]
    embeddings_buffer = []

    Path(args.output_path).mkdir(parents=True, exist_ok=True)  # Creates the output folder in case it does not exists
    for file in Path(args.content).glob('*'):
        with torch.no_grad():
            filename = args.content + '/' + file.name
            basename = os.path.basename(filename).split('.')[0]
            print('Processing ' + os.path.basename(filename) + ' with vtoonify_' + args.backbone[0])

            # Preprocess Image
            cropname, savename, sum_savename, frame = pre_processingImage(args, filename, basename, landmarkpredictor)

            # Encode Image
            s_w = encode_face_img(device, frame, landmarkpredictor)

            # Stylize pSp image
            if args.psp_style:
                print('Stylizing image with pSp')
                s_w = applyExstyle(s_w, exstyle, latent_mask)

            embeddings_buffer.append(torch.squeeze(s_w))
            window_slide()

            # Compute Mean
            s_w = pSpFeaturesBufferMean(embeddings_buffer)

            # Update VToonify Frame to mean face
            original_frame_size = frame.shape[:2]
            frame = decodeFeaturesToImg(s_w, vtoonify)

            if args.skip_vtoonify:
                cv2.imwrite(sum_savename, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                continue

            print('Using VToonify to stylize image')
            # Resize frame to save memory
            frame = cv2.resize(frame, original_frame_size)

            # Compute VToonify Features
            s_w, inputs = processingStyle(device, frame, s_w)

            # Process Image with VToonify
            y_tilde = vtoonify(inputs, s_w.repeat(inputs.size(0), 1, 1), d_s=args.style_degree)
            y_tilde = torch.clamp(y_tilde, -1, 1)

            # Save Output Image
            cv2.imwrite(cropname, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            save_image(y_tilde[0].cpu(), savename)
