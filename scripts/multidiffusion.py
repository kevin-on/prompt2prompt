import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))


import torch
from diffusers import StableDiffusionPipeline  # type: ignore

import ptp_utils  # FIXME: This import is a temporary workaround for circular dependency issues
from attn_controller import AttentionControlConfig
from attn_multidiffusion import generate
from pil_utils import save_image

LOW_RESOURCE = False
NUM_DIFFUSION_STEPS = 50
GUIDANCE_SCALE = 7.5
MAX_NUM_WORDS = 77
SEED = 8888
IMAGE_SIZE = 512


def main():
    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    ldm_stable = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
    ).to(device)
    attn_config = AttentionControlConfig(
        device=device,
        tokenizer=ldm_stable.tokenizer,
        max_num_words=MAX_NUM_WORDS,
        low_resource=LOW_RESOURCE,
    )

    alphas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    for alpha in alphas:
        images = generate(
            model=ldm_stable,
            prompt="A full view of the Eiffel Tower standing tall",
            height=512,
            width=1024,
            config=attn_config,
            num_inference_steps=NUM_DIFFUSION_STEPS,
            guidance_scale=GUIDANCE_SCALE,
            alpha=alpha,
            generator=torch.Generator().manual_seed(SEED),
            baseline=False,
        )
        save_image(
            images[0], f"results/multidiffusion/eiffel_tower_alpha{alpha:.1f}.png"
        )

    images = generate(
        model=ldm_stable,
        prompt="A full view of the Eiffel Tower standing tall",
        height=512,
        width=1024,
        config=attn_config,
        num_inference_steps=NUM_DIFFUSION_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        alpha=alpha,
        generator=torch.Generator().manual_seed(SEED),
        baseline=True,
    )
    save_image(images[0], "results/multidiffusion/eiffel_tower_baseline.png")


if __name__ == "__main__":
    main()
