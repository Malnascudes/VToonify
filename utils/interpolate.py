import math
import numpy as np
from typing import *

import torch
import PIL

def interpolate(
    latents_list: List[Union[torch.Tensor, np.ndarray]],
    duration_list: List[float],
    interpolation_type: str = "sinusoidal",  #sinusoidal | linear
    loop: bool = True,
    FPS: bool = 25,
    **kwargs,
) -> List[PIL.Image.Image]:
    """
    Returns a list of PIL images corresponding to the 
    generations produced by interpolating the values
    from `latents_list`.
    Args:
        latents_list (List[torch.Tensor]): list of VQVAE 
            intermediate embeddings.
        duration_list (List[float]): list with the duration of
            the interpolation between each consecutive tensor. 
            The last value represent the duration between the 
            last tensor and the initial.
        interpolation_type (str): either "sinusoidal" or "linear".
    Returns:
        List[PIL.Image.Image]: list of the resulting generated images.
    """
    z_logits_list = latents_list

    for idx, (z_logits, duration) in enumerate(zip(z_logits_list, duration_list)):
        if idx == len(z_logits_list) - 1 and not loop:
            break

        num_steps = int(duration * FPS)
        z_logits_1 = z_logits
        z_logits_2 = z_logits_list[(idx + 1) % len(z_logits_list)]

        for step in range(num_steps):
            if step == num_steps - 1 and num_steps > 1:
                weight = 1

            else:
                if interpolation_type == "linear":
                    if num_steps - 1 > 0:
                        weight = step / (num_steps - 1)
                    else:
                        weight = 0

                else:
                    weight = math.sin(1.5708 * step / num_steps)**2

            z_logits = weight * z_logits_2 + (1 - weight) * z_logits_1

            if isinstance(z_logits, np.ndarray):
                z_logits = torch.tensor(z_logits, device='cuda:0', dtype=torch.float32).unsqueeze(0)
            elif isinstance(z_logits, torch.Tensor):
                z_logits = z_logits.unsqueeze(0)

            torch.cuda.empty_cache()
            yield z_logits

    # return gen_img_list