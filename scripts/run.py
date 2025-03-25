import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from typing import List, Optional

import numpy as np
import torch
from diffusers import DiffusionPipeline  # type: ignore
from diffusers import StableDiffusionPipeline  # type: ignore

import ptp_utils
from attn_controller import (
    AttentionControl,
    AttentionControlConfig,
    AttentionRefine,
    AttentionReplace,
    AttentionReweight,
    AttentionStore,
    EmptyControl,
    LocalBlend,
    get_equalizer,
)
from utils import get_cross_attention

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

    # Visualizing cross attention
    visualize_cross_attention(model=ldm_stable, config=attn_config, latent=init_latent)

    # AttentionReplace
    replace_edit(model=ldm_stable, config=attn_config, latent=init_latent)

    # LocalBlend
    local_edit(model=ldm_stable, config=attn_config, latent=init_latent)

    # AttentionRefine
    refinement_edit(model=ldm_stable, config=attn_config, latent=init_latent)

    # AttentionReweight
    attention_reweight(model=ldm_stable, config=attn_config, latent=init_latent)


def run(
    model: DiffusionPipeline,
    prompts: List[str],
    controller: AttentionControl,
    latent: Optional[torch.Tensor] = None,
    run_baseline: bool = False,
    generator: Optional[torch.Generator] = None,
):
    images, latents = ptp_utils.text2image_ldm_stable(
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
        baseline_images, baseline_latents = ptp_utils.text2image_ldm_stable(
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


def visualize_cross_attention(
    model: DiffusionPipeline,
    config: AttentionControlConfig,
    latent: Optional[torch.Tensor] = None,
):
    generator = torch.Generator().manual_seed(SEED)

    prompts = ["A painting of a squirrel eating a burger"]
    controller = AttentionStore(config)
    result = run(
        model,
        prompts,
        controller,
        latent=latent,
        run_baseline=False,
        generator=generator,
    )
    cross_attention = get_cross_attention(
        prompts,
        config.tokenizer,
        controller,
        res=16,
        from_where=["up", "down"],
    )

    ptp_utils.save_images(result["images"], "results/cross_attention_image.png")
    ptp_utils.save_images(
        cross_attention,
        "results/cross_attention_map.png",
    )


def replace_edit(
    model: DiffusionPipeline,
    config: AttentionControlConfig,
    latent: Optional[torch.Tensor] = None,
):
    generator = torch.Generator().manual_seed(SEED)
    prompts = [
        "A painting of a squirrel eating a burger",
        "A painting of a lion eating a burger",
    ]

    controller = AttentionReplace(
        config=config,
        prompts=prompts,
        num_steps=NUM_DIFFUSION_STEPS,
        cross_replace_steps=0.8,
        self_replace_steps=0.4,
    )
    result = run(
        model,
        prompts,
        controller,
        latent=latent,
        run_baseline=True,
        generator=generator,
    )
    ptp_utils.save_images(
        np.concatenate([result["images"], result["baseline_images"]], axis=0),
        "results/squirrel_to_lion.png",
        num_rows=2,
    )

    # Use a different cross_replace_steps
    controller2 = AttentionReplace(
        config=config,
        prompts=prompts,
        num_steps=NUM_DIFFUSION_STEPS,
        cross_replace_steps={"default_": 1.0, "lion": 0.4},
        self_replace_steps=0.4,
    )
    result2 = run(
        model,
        prompts,
        controller2,
        latent=latent,
        run_baseline=False,
        generator=generator,
    )
    ptp_utils.save_images(
        result2["images"], "results/squirrel_to_lion_less_replace.png"
    )


def local_edit(
    model: DiffusionPipeline,
    config: AttentionControlConfig,
    latent: Optional[torch.Tensor] = None,
):
    generator = torch.Generator().manual_seed(SEED)

    # Preserve the burger
    prompts = [
        "A painting of a squirrel eating a burger",
        "A painting of a lion eating a burger",
    ]
    lb = LocalBlend(config=config, prompts=prompts, words=[["squirrel"], ["lion"]])
    controller = AttentionReplace(
        config=config,
        prompts=prompts,
        num_steps=NUM_DIFFUSION_STEPS,
        cross_replace_steps={"default_": 1.0, "lion": 0.4},
        self_replace_steps=0.4,
        local_blend=lb,
    )
    result = run(
        model,
        prompts,
        controller,
        latent=latent,
        run_baseline=True,
        generator=generator,
    )
    ptp_utils.save_images(
        np.concatenate([result["images"], result["baseline_images"]], axis=0),
        "results/squirrel_to_lion_local_blend.png",
        num_rows=2,
    )

    # Preserve the squirrel
    prompts = [
        "A painting of a squirrel eating a burger",
        "A painting of a squirrel eating a lasagne",
    ]
    lb = LocalBlend(config=config, prompts=prompts, words=[["burger"], ["lasagne"]])
    controller = AttentionReplace(
        config=config,
        prompts=prompts,
        num_steps=NUM_DIFFUSION_STEPS,
        cross_replace_steps={"default_": 1.0, "lasagne": 0.2},
        self_replace_steps=0.4,
        local_blend=lb,
    )
    result = run(
        model,
        prompts,
        controller,
        latent=latent,
        run_baseline=True,
        generator=generator,
    )
    ptp_utils.save_images(
        np.concatenate([result["images"], result["baseline_images"]], axis=0),
        "results/burger_to_lasagne_local_blend.png",
        num_rows=2,
    )


def refinement_edit(
    model: DiffusionPipeline,
    config: AttentionControlConfig,
    latent: Optional[torch.Tensor] = None,
):
    generator = torch.Generator().manual_seed(SEED)

    prompts = [
        "A painting of a squirrel eating a burger",
        "A neoclassical painting of a squirrel eating a burger",
    ]
    controller = AttentionRefine(
        config=config,
        prompts=prompts,
        num_steps=NUM_DIFFUSION_STEPS,
        cross_replace_steps=0.5,
        self_replace_steps=0.2,
    )
    result = run(
        model,
        prompts,
        controller,
        latent=latent,
        run_baseline=True,
        generator=generator,
    )
    ptp_utils.save_images(
        np.concatenate([result["images"], result["baseline_images"]], axis=0),
        "results/squirrel_to_neoclassical.png",
        num_rows=2,
    )

    # Fall mountain
    prompts = [
        "a photo of a house on a mountain",
        "a photo of a house on a mountain at fall",
    ]
    controller = AttentionRefine(
        config=config,
        prompts=prompts,
        num_steps=NUM_DIFFUSION_STEPS,
        cross_replace_steps=0.8,
        self_replace_steps=0.4,
    )
    result = run(
        model,
        prompts,
        controller,
        latent=latent,
        run_baseline=True,
        generator=generator,
    )
    ptp_utils.save_images(
        np.concatenate([result["images"], result["baseline_images"]], axis=0),
        "results/mountain_at_fall.png",
        num_rows=2,
    )

    # Winter mountain
    prompts = [
        "a photo of a house on a mountain",
        "a photo of a house on a mountain at winter",
    ]
    controller = AttentionRefine(
        config=config,
        prompts=prompts,
        num_steps=NUM_DIFFUSION_STEPS,
        cross_replace_steps=0.8,
        self_replace_steps=0.4,
    )
    result = run(
        model,
        prompts,
        controller,
        latent=latent,
        run_baseline=True,
        generator=generator,
    )
    ptp_utils.save_images(
        np.concatenate([result["images"], result["baseline_images"]], axis=0),
        "results/mountain_at_winter.png",
        num_rows=2,
    )


def refinement_edit_with_local_blend(
    model: DiffusionPipeline,
    config: AttentionControlConfig,
    latent: Optional[torch.Tensor] = None,
):
    generator = torch.Generator().manual_seed(SEED)

    prompts = ["soup", "pea soup"]
    lb = LocalBlend(config=config, prompts=prompts, words=[["soup"], ["soup"]])
    controller = AttentionRefine(
        config=config,
        prompts=prompts,
        num_steps=NUM_DIFFUSION_STEPS,
        cross_replace_steps=0.8,
        self_replace_steps=0.4,
        local_blend=lb,
    )
    result = run(
        model,
        prompts,
        controller,
        latent=latent,
        run_baseline=True,
        generator=generator,
    )
    ptp_utils.save_images(
        np.concatenate([result["images"], result["baseline_images"]], axis=0),
        "results/soup_to_pea_soup_local_blend.png",
        num_rows=2,
    )


def attention_reweight(
    model: DiffusionPipeline,
    config: AttentionControlConfig,
    latent: Optional[torch.Tensor] = None,
):
    generator = torch.Generator().manual_seed(SEED)

    # Example 1 - pay 3 times more attention to the word "smiling"
    prompts = ["a smiling bunny doll"] * 2
    equalizer = get_equalizer(prompts[1], ("smiling",), (5,), config=config)
    controller_refine = AttentionReweight(
        config=config,
        prompts=prompts,
        num_steps=NUM_DIFFUSION_STEPS,
        cross_replace_steps=0.8,
        self_replace_steps=0.4,
        equalizer=equalizer,
    )
    result = run(
        model,
        prompts,
        controller_refine,
        latent=latent,
        run_baseline=True,
        generator=generator,
    )
    ptp_utils.save_images(
        np.concatenate([result["images"], result["baseline_images"]], axis=0),
        "results/smiling_bunny_doll.png",
        num_rows=2,
    )

    # Example 2 - pay less attention to the word "pink"
    prompts = ["pink bear riding a bicycle"] * 2
    equalizer = get_equalizer(prompts[1], ("pink",), (-1,), config=config)
    lb = LocalBlend(config=config, prompts=prompts, words=[["bicycle"], ["bicycle"]])
    controller_refine = AttentionReweight(
        config=config,
        prompts=prompts,
        num_steps=NUM_DIFFUSION_STEPS,
        cross_replace_steps=0.8,
        self_replace_steps=0.4,
        equalizer=equalizer,
        local_blend=lb,
    )
    result = run(
        model,
        prompts,
        controller_refine,
        latent=latent,
        run_baseline=True,
        generator=generator,
    )
    ptp_utils.save_images(
        np.concatenate([result["images"], result["baseline_images"]], axis=0),
        "results/pink_bear_bicycle.png",
        num_rows=2,
    )

    # Example 3 - pay 3 times more attention to the word "croutons"
    prompts = ["soup", "pea soup with croutons"]
    lb = LocalBlend(config=config, prompts=prompts, words=[["soup"], ["soup"]])
    controller_refine = AttentionRefine(
        config=config,
        prompts=prompts,
        num_steps=NUM_DIFFUSION_STEPS,
        cross_replace_steps=0.8,
        self_replace_steps=0.4,
        local_blend=lb,
    )
    result = run(
        model,
        prompts,
        controller_refine,
        latent=latent,
        run_baseline=True,
        generator=generator,
    )
    ptp_utils.save_images(
        np.concatenate([result["images"], result["baseline_images"]], axis=0),
        "results/soup_to_pea_soup_croutons.png",
        num_rows=2,
    )

    equalizer = get_equalizer(prompts[1], ("croutons",), (3,), config=config)
    controller_reweight = AttentionReweight(
        config=config,
        prompts=prompts,
        num_steps=NUM_DIFFUSION_STEPS,
        cross_replace_steps=0.8,
        self_replace_steps=0.4,
        equalizer=equalizer,
        local_blend=lb,
        controller=controller_refine,
    )
    result = run(
        model,
        prompts,
        controller_reweight,
        latent=latent,
        run_baseline=True,
        generator=generator,
    )
    ptp_utils.save_images(
        np.concatenate([result["images"], result["baseline_images"]], axis=0),
        "results/soup_to_pea_soup_croutons_reweight.png",
        num_rows=2,
    )

    # Example 4 - pay 10 times more attention to the word "fried"
    prompts = ["potatos", "fried potatos"]
    lb = LocalBlend(config=config, prompts=prompts, words=[["potatos"], ["potatos"]])
    controller_refine = AttentionRefine(
        config=config,
        prompts=prompts,
        num_steps=NUM_DIFFUSION_STEPS,
        cross_replace_steps=0.8,
        self_replace_steps=0.4,
        local_blend=lb,
    )
    result = run(
        model,
        prompts,
        controller_refine,
        latent=latent,
        run_baseline=True,
        generator=generator,
    )
    ptp_utils.save_images(
        np.concatenate([result["images"], result["baseline_images"]], axis=0),
        "results/potato_to_fried_potato.png",
        num_rows=2,
    )

    equalizer = get_equalizer(prompts[1], ("fried",), (10,), config=config)
    controller_reweight = AttentionReweight(
        config=config,
        prompts=prompts,
        num_steps=NUM_DIFFUSION_STEPS,
        cross_replace_steps=0.8,
        self_replace_steps=0.4,
        equalizer=equalizer,
        local_blend=lb,
        controller=controller_refine,
    )
    result = run(
        model,
        prompts,
        controller_reweight,
        latent=latent,
        run_baseline=True,
        generator=generator,
    )
    ptp_utils.save_images(
        np.concatenate([result["images"], result["baseline_images"]], axis=0),
        "results/potato_to_fried_potato_reweight.png",
        num_rows=2,
    )


if __name__ == "__main__":
    main()
