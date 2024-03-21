import argparse
import sys

from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler
import numpy as np
from PIL import Image
import torch

# REVIEW
_DEFAULT_DEVICE = (
    'cuda' if torch.cuda.is_available()
    else 'mps' if torch.backends.mps.is_available()
    else 'cpu'
)


class TextToObjectImage:
    def __init__(
        self,
        device=_DEFAULT_DEVICE,
        model='Lykon/dreamshaper-8',
        cn_model='lllyasviel/control_v11p_sd15_normalbae',
    ):
        controlnet = ControlNetModel.from_pretrained(cn_model, torch_dtype=torch.float16, variant='fp16')

        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            model, controlnet=controlnet, torch_dtype=torch.float16, variant='fp16',
            safety_checker=None,
        )
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe = self.pipe.to(device)

    def generate(self, desc: str, steps: int, control_image: Image):
        return self.pipe(
            prompt=f'{desc}, front and back view, 180, reverse, 3D rendering, high quality 4K, flat',
            negative_prompt='lighting, shadows, grid, dark, mesh',
            num_inference_steps=steps,
            num_images_per_prompt=1,
            image=control_image,
            width=control_image.width,
            height=control_image.height,
        ).images[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('desc', help='Short description of desired model appearance')
    parser.add_argument('depth_img', help='Depth control image')
    parser.add_argument('output_path', help='Path for generated image')
    parser.add_argument(
        '--image-model',
        help='SD 1.5-based model for texture image gen',
        default='Lykon/dreamshaper-8',
    )
    parser.add_argument('--steps', type=int, default=12)
    parser.add_argument(
        '--device',
        default=_DEFAULT_DEVICE,
        type=str,
        help='Device to prefer. Default: try to auto-detect from platform (CUDA, Metal)'
    )
    args = parser.parse_args()

    t2i = TextToObjectImage(args.device, args.image_model)
    t2i.generate(args.desc, args.steps, Image.open(args.depth_img)).save(args.output_path)
