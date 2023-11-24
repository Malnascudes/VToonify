import os
import cv2
import numpy as np
import torch
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from ts.context import Context
from main import VToonifyHandler


if __name__ == "__main__":
    print('Testing VToonifyHandler.handle() method')
    test_images = ['./data/077436.jpg', './data/ILip77SbmOE.jpg']
    test_output_path = './tests'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_dir = "pretrained_models"
    model_name = "psp_ffhq_encode.pt"
    manifest = {
        'models': {
            'vtoonify': './checkpoint/vtoonify_d_cartoon/vtoonify_s_d_c.pt',
            'faceparsing': './checkpoint/faceparsing.pth',
            'style_encoder': './checkpoint/encoder.pt',
            'backbone': 'dualstylegan',
            'exstyle': './checkpoint/vtoonify_d_cartoon/exstyle_code_custom.npy',
            'style_id': 1,
        }
    }
    context = Context(model_dir=model_dir, model_name="vtoonify", manifest=manifest,batch_size=1,gpu=0,mms_version="1.0.0")

    print('Loading VToonify Model')
    vtoonify_handler = VToonifyHandler()
    vtoonify_handler.initialize(context)
    vtoonify_handler.FPS = 4

    print('Loaded models successfully!')

    # Constants and variables
    scale = 1
    kernel_1d = np.array([[0.125], [0.375], [0.375], [0.125]])

    for image_path in test_images:    
        basename = os.path.basename(image_path)    
        print('Testing with ' + basename)
        animation_frames = vtoonify_handler.handle(
            image_path,
            scale_image=False,
            padding=[200, 200, 200, 200],
            latent_mask=[],
            style_degree=0.5,
            skip_vtoonify=False,
        )

        image_test_output_path = Path(test_output_path, basename)
        Path(image_test_output_path).mkdir(parents=True, exist_ok=True)  # Creates the output folder in case it does not exists
        for i, frame in enumerate(animation_frames):
            save_path = Path(image_test_output_path, f'{i}.jpg')
            cv2.imwrite(str(save_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)) # save only last frame for test
