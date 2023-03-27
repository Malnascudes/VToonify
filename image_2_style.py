import cv2
import os
import numpy as np
import cv2
import dlib
import torch
from model.vtoonify import VToonify
from model.encoder.align_all_parallel import align_face
from util import save_image, load_image, visualize, load_psp_standalone, get_video_crop_parameter, tensor2cv2
from torchvision import transforms
import argparse


def create_arg_parser():
    parser = argparse.ArgumentParser(description="Style Transfer")
    parser.add_argument("--style_image_path", required=True, type=str, help="path of the image to convert to a style")
    parser.add_argument("--exstyle_path", required=True, type=str, help="path of the extrinsic style code")
    parser.add_argument("--style_encoder_path", type=str, default='./checkpoint/encoder.pt', help="path of the style encoder")
    parser.add_argument("--ckpt", type=str, default='./checkpoint/vtoonify_d_cartoon/vtoonify_s_d.pt', help="path of the saved model")
    parser.add_argument("--backbone", type=str, default='dualstylegan', help="dualstylegan | toonify")
    parser.add_argument("--cpu", action="store_true", help="if true, only use cpu")
    return parser

if __name__ == "__main__":
    parser = create_arg_parser()
    args = parser.parse_args()
    args
    style_image_path = args.style_image_path
    exstyle_path = args.exstyle_path
    style_encoder_path = args.style_encoder_path
    vtoonify_ckpt = args.ckpt
    device = 'cpu' if args.cpu else 'cuda'
    backbone = args.backbone
    # style_image_path = '/Users/carlesanton/repos/matriu/ideal/Datasets/styles/elf/fairy-krea-2.png'
    # exstyle_path = './checkpoint/arcane/exstyle_code_custom.npy'
    # style_encoder_path = './checkpoint/encoder.pt'
    # vtoonify_ckpt = './checkpoint/arcane/vtoonify_s_d.pt'
    # device = 'cpu'
    # backbone = 'dualstylegan'

    landmark_modelname = './checkpoint/shape_predictor_68_face_landmarks.dat'
    print('Loading Landmark Predictor')
    landmarkpredictor = dlib.shape_predictor(landmark_modelname)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5,0.5,0.5]),
    ])
    print('Loading PSP Encoder')
    pspencoder = load_psp_standalone(style_encoder_path, device)    
    print('Loading VToonify')
    vtoonify = VToonify(backbone = backbone)
    vtoonify.load_state_dict(torch.load(vtoonify_ckpt, map_location=lambda storage, loc: storage)['g_ema'])
    vtoonify.to(device)

    image_name = style_image_path.split('/')[-1]
    print('Loading Image')
    frame = cv2.imread(style_image_path)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    print('Aligning Face')
    try:
        I = align_face(frame, landmarkpredictor)
        I = transform(I).unsqueeze(dim=0).to(device)
    except:
        print('Unable to align face, using raw image')
        I = transform(frame).unsqueeze(dim=0).to(device)
    print('Encoding Image')
    s_w = pspencoder(I)
    # also store the embeding without zplus2wplus to compare results
    new_style_embeding_no_zplus2wplus = s_w.cpu().detach().numpy()
    s_w = vtoonify.zplus2wplus(s_w)
    new_style_embeding = s_w.cpu().detach().numpy()

    if os.path.exists(exstyle_path):
        exstyles = np.load(exstyle_path, allow_pickle='TRUE').item()
        exstyles[image_name] = new_style_embeding
        exstyles[f'no-zplus2wplus-{image_name}'] = new_style_embeding_no_zplus2wplus
    else:
        exstyles = {image_name: new_style_embeding, f'no-zplus2wplus-{image_name}': new_style_embeding_no_zplus2wplus}

    print(f'Saving new style encoding from {image_name} to position {len(exstyles.items())-1} of {exstyle_path}')
    np.save(exstyle_path, exstyles)