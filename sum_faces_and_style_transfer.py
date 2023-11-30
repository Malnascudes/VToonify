import os
from pathlib import Path
#os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import argparse
import numpy as np
import cv2
import dlib
import torch
from torchvision import transforms
import torch.nn.functional as F
from tqdm import tqdm
from model.vtoonify import VToonify
from model.bisenet.bisnet_model import BiSeNet
from model.encoder.align_all_parallel import align_face
from util import save_image, load_image, visualize, load_psp_standalone, get_video_crop_parameter, tensor2cv2

SLIDING_WINDOW_SIZE = 9

class TestOptions():
    def __init__(self):

        self.parser = argparse.ArgumentParser(description="Style Transfer")
        self.parser.add_argument("--content", type=str, default='./data', help="path of the folder with the content image/video")
        self.parser.add_argument("--style_id", type=int, default=26, help="the id of the style image")
        self.parser.add_argument("--style_degree", type=float, default=0.5, help="style degree for VToonify-D")
        self.parser.add_argument("--color_transfer", action="store_true", help="transfer the color of the style")
        self.parser.add_argument("--ckpt", type=str, default='./checkpoint/vtoonify_d_cartoon/vtoonify_s_d.pt', help="path of the saved model")
        self.parser.add_argument("--output_path", type=str, default='./output/', help="path of the output images")
        self.parser.add_argument("--scale_image", action="store_true", help="resize and crop the image to best fit the model")
        self.parser.add_argument("--style_encoder_path", type=str, default='./checkpoint/encoder.pt', help="path of the style encoder")
        self.parser.add_argument("--exstyle_path", type=str, default=None, help="path of the extrinsic style code")
        self.parser.add_argument("--faceparsing_path", type=str, default='./checkpoint/faceparsing.pth', help="path of the face parsing model")
        self.parser.add_argument("--cpu", action="store_true", help="if true, only use cpu")
        self.parser.add_argument("--backbone", type=str, default='dualstylegan', help="dualstylegan | toonify")
        self.parser.add_argument("--padding", type=int, nargs=4, default=[200,200,200,200], help="left, right, top, bottom paddings to the face center")
        
    def parse(self):
        self.opt = self.parser.parse_args()
        if self.opt.exstyle_path is None:
            self.opt.exstyle_path = os.path.join(os.path.dirname(self.opt.ckpt), 'exstyle_code.npy')
        args = vars(self.opt)
        print('Load options')
        for name, value in sorted(args.items()):
            print('%s: %s' % (str(name), str(value)))
        return self.opt

def window_slide():
    if len(embeddings_buffer) > SLIDING_WINDOW_SIZE:
        embeddings_buffer.pop(0)


def pre_processingImage(args, filename, basename, landmarkpredictor):
    cropname = os.path.join(args.output_path, basename + '_input.jpg')
    savename = os.path.join(args.output_path, basename + '_vtoonify_' +  args.backbone[0] + '.jpg')
    sum_savename = os.path.join(args.output_path, basename + '_vtoonify_SUM' +  args.backbone[0] + '.jpg')

    frame = cv2.imread(filename)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # We detect the face in the image, and resize the image so that the eye distance is 64 pixels.
    # Centered on the eyes, we crop the image to almost 400x400 (based on args.padding).
    if args.scale_image:
        paras = get_video_crop_parameter(frame, landmarkpredictor, args.padding)
        if paras is not None:
            h,w,top,bottom,left,right,scale = paras
            H, W = int(bottom-top), int(right-left)
            # for HR image, we apply gaussian blur to it to avoid over-sharp stylization results
            if scale <= 0.75:
                frame = cv2.sepFilter2D(frame, -1, kernel_1d, kernel_1d)
            if scale <= 0.375:
                frame = cv2.sepFilter2D(frame, -1, kernel_1d, kernel_1d)
            frame = cv2.resize(frame, (w, h))[top:bottom, left:right]

    return cropname, savename, sum_savename, frame

def encoding(I):
    s_w = pspencoder(I)
    # if previous_embedding is not None:
    #     vector_encoded = (vector_encoded + previous_embedding)/2  # SUM OPERATION? HERE?
    s_w = vtoonify.zplus2wplus(s_w)
    if vtoonify.backbone == 'dualstylegan':
        if args.color_transfer:
            s_w = exstyle
        else:
            s_w[:,:7] = exstyle[:,:7]

    return s_w

def styling(device, frame, s_w):
    x = transform(frame).unsqueeze(dim=0).to(device)
    # parsing network works best on 512x512 images, so we predict parsing maps on upsmapled frames
    # followed by downsampling the parsing maps
    x_p = F.interpolate(parsingpredictor(2*(F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)))[0], scale_factor=0.5, recompute_scale_factor=False).detach()
    # we give parsing maps lower weight (1/16)
    inputs = torch.cat((x, x_p/16.), dim=1)
    # d_s has no effect when backbone is toonify
    y_tilde = vtoonify(inputs, s_w.repeat(inputs.size(0), 1, 1), d_s = args.style_degree)
    y_tilde = torch.clamp(y_tilde, -1, 1)

    return y_tilde

def processingImage(device, frame, landmarkpredictor):
    with torch.no_grad():
            I = align_face(frame, landmarkpredictor)
            I = transform(I).unsqueeze(dim=0).to(device)

            s_w = encoding(I)
            y_tilde = styling(device, frame, s_w)

            embeddings_buffer.append(s_w)

            window_slide()

            if len(embeddings_buffer) > 1:
                add_embedding = torch.stack(embeddings_buffer)
                sum_embedding = torch.mean(add_embedding, 0)
                y_tilde_sum = styling(device, frame, sum_embedding)
                return y_tilde, y_tilde_sum

            return y_tilde, None

if __name__ == "__main__":

    # PROCESSING THE INPUT VALUES
    parser = TestOptions()
    args = parser.parse()
    print('*'*98)
    
    device = "cpu" if args.cpu else "cuda"
    # # Force the code to be runned in my M2  chip all the time
    # device = "cpu"
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5,0.5,0.5]),
        ])
    
    vtoonify = VToonify(backbone = args.backbone)
    vtoonify.load_state_dict(torch.load(args.ckpt, map_location=lambda storage, loc: storage)['g_ema'])
    vtoonify.to(device)

    parsingpredictor = BiSeNet(n_classes=19)
    parsingpredictor.load_state_dict(torch.load(args.faceparsing_path, map_location=lambda storage, loc: storage))
    parsingpredictor.to(device).eval()

    modelname = './checkpoint/shape_predictor_68_face_landmarks.dat'
    if not os.path.exists(modelname):
        import wget, bz2
        wget.download('http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2', modelname+'.bz2')
        zipfile = bz2.BZ2File(modelname+'.bz2')
        data = zipfile.read()
        open(modelname, 'wb').write(data) 
    landmarkpredictor = dlib.shape_predictor(modelname)

    pspencoder = load_psp_standalone(args.style_encoder_path, device)    

    if args.backbone == 'dualstylegan':
        exstyles = np.load(args.exstyle_path, allow_pickle='TRUE').item()
        stylename = list(exstyles.keys())[args.style_id]
        exstyle = torch.tensor(exstyles[stylename]).to(device)
        with torch.no_grad():  
            exstyle = vtoonify.zplus2wplus(exstyle)

    embeddings_buffer = []

    print('Load models successfully!')

    files = Path(args.content).glob('*')
    for file in files:
        filename = args.content + "/" + file.name
        basename = os.path.basename(filename).split('.')[0]
        scale = 1
        kernel_1d = np.array([[0.125],[0.375],[0.375],[0.125]])

        Path(args.output_path).mkdir(parents=True, exist_ok=True)  # Creates the output folder in case it does not exists
        print('Processing ' + os.path.basename(filename) + ' with vtoonify_' + args.backbone[0])

        # PROCESSING THE IMAGES

        cropname, savename, sum_savename, frame = pre_processingImage(args, filename, basename, landmarkpredictor)
    
        y_tilde, y_tilde_sum = processingImage(device, frame, landmarkpredictor)
        
        cv2.imwrite(cropname, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        save_image(y_tilde[0].cpu(), savename)

        if y_tilde_sum is not None:
            # cv2.imwrite(cropname, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            save_image(y_tilde_sum[0].cpu(), sum_savename)
        
        print('Transfer style successfully!')
