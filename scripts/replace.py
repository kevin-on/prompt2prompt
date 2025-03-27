import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from typing import List, Optional

import numpy as np
import torch
from diffusers import DiffusionPipeline  # type: ignore
from diffusers import StableDiffusionPipeline  # type: ignore

from attn_controller import (
    AttentionControl,
    AttentionControlConfig,
    AttentionReplace,
    EmptyControl,
)
from pil_utils import create_image_grid, save_image
from ptp_utils import text2image_ldm_stable

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
    init_latent = torch.randn(
        (1, ldm_stable.unet.config.in_channels, IMAGE_SIZE // 8, IMAGE_SIZE // 8),
        generator=torch.Generator().manual_seed(SEED),
    )

    prompts = [
        "A painting of a squirrel eating a burger",
        "A painting of a lion eating a burger",
    ]
    test_replace_steps(
        prompts=prompts, model=ldm_stable, config=attn_config, latent=init_latent
    )


class AttentionReplaceInEveryLayer(AttentionReplace):
    """
    A variant of AttentionReplace that replaces self-attention in all layers.

    Unlike the original AttentionReplace which only affects layers with spatial
    dimensions of 16x16 or smaller, this class applies the replacement to every
    layer regardless of its dimensions.
    """

    def replace_self_attention(
        self, attn_base: torch.Tensor, att_replace: torch.Tensor
    ) -> torch.Tensor:
        return attn_base.unsqueeze(0).expand(att_replace.shape[0], *attn_base.shape)


def test_replace_steps(
    prompts: List[str],
    model: DiffusionPipeline,
    config: AttentionControlConfig,
    latent: Optional[torch.Tensor] = None,
):
    generator = torch.Generator().manual_seed(SEED)

    original_image = None
    replaced_images = []
    self_replace_steps = np.linspace(0.0, 1.0, 6).tolist()
    cross_replace_steps = np.linspace(0.0, 1.0, 6).tolist()

    for self_replace_step in self_replace_steps:
        batch_images = []
        for cross_replace_step in cross_replace_steps:
            controller = AttentionReplace(
                # controller = AttentionReplaceInEveryLayer(
                config=config,
                prompts=prompts,
                num_steps=NUM_DIFFUSION_STEPS,
                cross_replace_steps=cross_replace_step,
                self_replace_steps=self_replace_step,
            )
            result = run(
                model,
                prompts,
                controller,
                latent=latent,
                run_baseline=False,
                generator=generator,
            )
            if original_image is None:
                original_image = result["images"][0]
            batch_images.append(result["images"][1])
        replaced_images.append(batch_images)

    save_image(
        create_image_grid(
            replaced_images,
            num_rows=len(self_replace_steps),
            num_cols=len(cross_replace_steps),
            row_descs=[
                f"self_replace_steps: {self_replace_step:.2f}"
                for self_replace_step in self_replace_steps
            ],
            col_descs=[
                f"cross_replace_steps: {cross_replace_step:.2f}"
                for cross_replace_step in cross_replace_steps
            ],
        ),
        "results/replace/self-replace-every-layer_grid.png",
    )
    if original_image is not None:
        save_image(
            original_image,
            "results/replace/self-replace-every-layer_original.png",
        )


def run(
    model: DiffusionPipeline,
    prompts: List[str],
    controller: AttentionControl,
    latent: Optional[torch.Tensor] = None,
    run_baseline: bool = False,
    generator: Optional[torch.Generator] = None,
):
    images, latents = text2image_ldm_stable(
        model,
        prompts,
        controller,
        latent=latent,
        num_inference_steps=NUM_DIFFUSION_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        generator=generator,
        low_resource=LOW_RESOURCE,
    )

    baseline_images = None
    baseline_latents = None

    if run_baseline:
        baseline_images, baseline_latents = text2image_ldm_stable(
            model,
            prompts,
            EmptyControl(config=controller.config),
            latent=latent,
            num_inference_steps=NUM_DIFFUSION_STEPS,
            guidance_scale=GUIDANCE_SCALE,
            generator=generator,
            low_resource=LOW_RESOURCE,
        )

    return {
        "images": images,
        "latents": latents,
        "baseline_images": baseline_images,
        "baseline_latents": baseline_latents,
    }


if __name__ == "__main__":
    main()
