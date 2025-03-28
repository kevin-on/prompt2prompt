# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from typing import Dict, List, Literal, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from diffusers import DiffusionPipeline  # type: ignore
from PIL import Image
from tqdm import tqdm
from transformers import CLIPTokenizer

from attn_controller import AttentionControl
from attn_processor import Prompt2PromptAttnProcessor


def add_text_to_image(
    image: np.ndarray,
    text: str,
    position: Literal["top", "bottom"] = "bottom",
    offset_ratio: float = 0.15,
    text_color: Tuple[int, int, int] = (0, 0, 0),
    font_scale: float = 1.0,
    font_thickness: int = 2,
    bg_color: Tuple[int, int, int] = (255, 255, 255),
    font_face: int = cv2.FONT_HERSHEY_SIMPLEX,
):
    """Add text above or below an image with customizable parameters.

    Args:
        image: Input image as numpy array
        text: Text to add under the image
        text_color: RGB color tuple for text
        font_scale: Scale factor for font size
        font_thickness: Thickness of font strokes
        offset_ratio: Ratio of image height to use as offset for text
        bg_color: RGB color tuple for background
        font_face: OpenCV font face to use
        position: Where to place text - either "top" or "bottom"
    """
    if image.ndim != 3:
        raise ValueError(f"Invalid image shape: {image.shape}")

    if position not in ["top", "bottom"]:
        raise ValueError("Position must be either 'top' or 'bottom'")

    h, w, c = image.shape
    offset = int(h * offset_ratio)

    new_img = np.ones((h + offset, w, c), dtype=np.uint8) * np.array(
        bg_color, dtype=np.uint8
    )

    textsize = cv2.getTextSize(text, font_face, font_scale, font_thickness)[0]
    text_x = (w - textsize[0]) // 2

    if position == "bottom":
        new_img[:h] = image
        text_y = h + (offset + textsize[1]) // 2
    else:  # position == "top"
        new_img[offset:] = image
        text_y = (offset + textsize[1]) // 2

    cv2.putText(
        new_img,
        text,
        (text_x, text_y),
        font_face,
        font_scale,
        text_color,
        font_thickness,
    )
    return new_img


def save_images(
    images: Union[List[np.ndarray], np.ndarray],
    path: str,
    num_rows: int = 1,
    offset_ratio: float = 0.02,
):
    if isinstance(images, list):
        num_empty = len(images) % num_rows
    elif isinstance(images, np.ndarray):
        if images.ndim == 4:
            num_empty = images.shape[0] % num_rows
        elif images.ndim == 3:
            images = [images]
            num_empty = 0
        else:
            raise ValueError(f"Invalid image shape: {images.shape}")

    empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255
    images = [image.astype(np.uint8) for image in images] + [empty_images] * num_empty
    num_items = len(images)

    h, w, c = images[0].shape
    offset = int(h * offset_ratio)
    num_cols = num_items // num_rows
    image_ = (
        np.ones(
            (
                h * num_rows + offset * (num_rows - 1),
                w * num_cols + offset * (num_cols - 1),
                3,
            ),
            dtype=np.uint8,
        )
        * 255
    )
    for i in range(num_rows):
        for j in range(num_cols):
            image_[
                i * (h + offset) : i * (h + offset) + h :,
                j * (w + offset) : j * (w + offset) + w,
            ] = images[i * num_cols + j]

    pil_img = Image.fromarray(image_)

    if os.path.dirname(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    pil_img.save(path)


def diffusion_step(
    model: DiffusionPipeline,
    controller: AttentionControl,
    latents: torch.Tensor,
    context: Union[torch.Tensor, List[torch.Tensor]],
    t: int,
    guidance_scale: Optional[float],
    low_resource: bool = False,
):
    if low_resource:
        noise_pred_uncond = model.unet(latents, t, encoder_hidden_states=context[0])[
            "sample"
        ]
        noise_prediction_text = model.unet(
            latents, t, encoder_hidden_states=context[1]
        )["sample"]
    else:
        latents_input = torch.cat([latents] * 2)
        noise_pred = model.unet(latents_input, t, encoder_hidden_states=context)[
            "sample"
        ]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)

    if guidance_scale is not None:
        noise_pred = noise_pred_uncond + guidance_scale * (
            noise_prediction_text - noise_pred_uncond
        )
    latents = model.scheduler.step(noise_pred, t, latents)["prev_sample"]
    latents = controller.step_callback(latents)
    return latents


def latent2image(vae, latents: torch.Tensor) -> np.ndarray:
    latents = 1 / 0.18215 * latents
    image = vae.decode(latents)["sample"]
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image * 255).astype(np.uint8)
    return image


def init_latent(
    latent: Optional[torch.Tensor],
    model: DiffusionPipeline,
    height: int,
    width: int,
    generator: Optional[torch.Generator],
    batch_size: int,
) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
    if latent is None:
        latent = torch.randn(
            (1, model.unet.config.in_channels, height // 8, width // 8),
            generator=generator,
        )
    latents = latent.expand(
        batch_size, model.unet.config.in_channels, height // 8, width // 8
    ).to(model.device)
    return latent, latents


@torch.no_grad()
def text2image_ldm(
    model: DiffusionPipeline,
    prompt: List[str],
    controller: AttentionControl,
    num_inference_steps: int = 50,
    guidance_scale: Optional[float] = 7.0,
    generator: Optional[torch.Generator] = None,
    latent: Optional[torch.Tensor] = None,
):
    register_attention_control(model, controller)
    height = width = 256
    batch_size = len(prompt)

    uncond_input = model.tokenizer(
        [""] * batch_size, padding="max_length", max_length=77, return_tensors="pt"
    )
    uncond_embeddings = model.bert(uncond_input.input_ids.to(model.device))[0]

    text_input = model.tokenizer(
        prompt, padding="max_length", max_length=77, return_tensors="pt"
    )
    text_embeddings = model.bert(text_input.input_ids.to(model.device))[0]
    latent, latents = init_latent(latent, model, height, width, generator, batch_size)
    context = torch.cat([uncond_embeddings, text_embeddings])

    model.scheduler.set_timesteps(num_inference_steps)
    for t in tqdm(model.scheduler.timesteps):
        latents = diffusion_step(model, controller, latents, context, t, guidance_scale)

    image = latent2image(model.vqvae, latents)

    return image, latent


@torch.no_grad()
def text2image_ldm_stable(
    model: DiffusionPipeline,
    prompts: List[str],
    controller: AttentionControl,
    num_inference_steps: int = 50,
    guidance_scale: Optional[float] = 7.5,
    generator: Optional[torch.Generator] = None,
    latent: Optional[torch.Tensor] = None,
    low_resource: bool = False,
):
    register_attention_control(model, controller)
    height = width = 512
    batch_size = len(prompts)

    text_input = model.tokenizer(
        prompts,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings: torch.Tensor = model.text_encoder(
        text_input.input_ids.to(model.device)
    )[0]
    max_length = text_input.input_ids.shape[-1]
    uncond_input = model.tokenizer(
        [""] * batch_size,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    uncond_embeddings: torch.Tensor = model.text_encoder(
        uncond_input.input_ids.to(model.device)
    )[0]

    context = [uncond_embeddings, text_embeddings]
    if not low_resource:
        context = torch.cat(context)
    latent, latents = init_latent(latent, model, height, width, generator, batch_size)

    # set timesteps
    # "offset" is depreacted since diffusers 0.4.0
    # extra_set_kwargs = {"offset": 1}
    model.scheduler.set_timesteps(num_inference_steps)
    for t in tqdm(model.scheduler.timesteps):
        latents = diffusion_step(
            model, controller, latents, context, t, guidance_scale, low_resource
        )

    image = latent2image(model.vae, latents)

    return image, latent


def register_attention_control(model: DiffusionPipeline, controller: AttentionControl):
    def register_recr(net_: torch.nn.Module, count: int, place_in_unet: str) -> int:
        if net_.__class__.__name__ == "Attention":
            net_.processor = Prompt2PromptAttnProcessor(controller, place_in_unet)  # type: ignore
            return count + 1
        elif hasattr(net_, "children"):
            for net__ in net_.children():
                count = register_recr(net__, count, place_in_unet)
        return count

    cross_att_count = 0
    sub_nets = model.unet.named_children()
    for net in sub_nets:
        if "down" in net[0]:
            cross_att_count += register_recr(net[1], 0, "down")
        elif "up" in net[0]:
            cross_att_count += register_recr(net[1], 0, "up")
        elif "mid" in net[0]:
            cross_att_count += register_recr(net[1], 0, "mid")

    controller.num_att_layers = cross_att_count


def get_word_inds(
    text: str, word_place: Union[str, int], tokenizer: CLIPTokenizer
) -> np.ndarray:
    split_text = text.split(" ")
    word_indices: List[int] = []
    if type(word_place) is str:
        word_indices = [i for i, word in enumerate(split_text) if word_place == word]
    elif type(word_place) is int:
        word_indices = [word_place]
    out = []
    if len(word_indices) > 0:
        words_encode = [
            tokenizer.decode([item]).strip("#") for item in tokenizer.encode(text)
        ][1:-1]
        cur_len, ptr = 0, 0

        for i in range(len(words_encode)):
            cur_len += len(words_encode[i])
            if ptr in word_indices:
                out.append(i + 1)
            if cur_len >= len(split_text[ptr]):
                ptr += 1
                cur_len = 0
    return np.array(out)


def update_alpha_time_word(
    alpha: torch.Tensor,
    bounds_or_end: Union[float, Tuple[float, float]],
    prompt_ind: int,
    word_inds: Optional[Union[torch.Tensor, np.ndarray]] = None,
) -> torch.Tensor:
    if isinstance(bounds_or_end, float):
        bounds = [0, bounds_or_end]
    else:
        assert isinstance(bounds_or_end, tuple)
        bounds = bounds_or_end
    start, end = int(bounds[0] * alpha.shape[0]), int(bounds[1] * alpha.shape[0])
    if word_inds is None:
        word_inds = torch.arange(alpha.shape[2])
    alpha[:start, prompt_ind, word_inds] = 0
    alpha[start:end, prompt_ind, word_inds] = 1
    alpha[end:, prompt_ind, word_inds] = 0
    return alpha


def get_time_words_attention_alpha(
    prompts: List[str],
    num_steps: int,
    cross_replace_steps: Union[
        float, Tuple[float, float], Dict[str, Union[float, Tuple[float, float]]]
    ],
    tokenizer: CLIPTokenizer,
    max_num_words: int = 77,
) -> torch.Tensor:
    steps_dict: Dict[str, Tuple[float, float]] = {}
    if isinstance(cross_replace_steps, float):
        steps_dict = {"default_": (0, cross_replace_steps)}
    elif isinstance(cross_replace_steps, tuple):
        steps_dict = {"default_": cross_replace_steps}
    elif isinstance(cross_replace_steps, dict):
        for key, value in cross_replace_steps.items():
            if isinstance(value, float):
                steps_dict[key] = (0, value)
            elif isinstance(value, tuple):
                steps_dict[key] = value
            else:
                raise ValueError(f"Invalid cross_replace_steps value: {value}")

    if "default_" not in steps_dict:
        steps_dict["default_"] = (0.0, 1.0)
    alpha_time_words = torch.zeros(num_steps + 1, len(prompts) - 1, max_num_words)
    for i in range(len(prompts) - 1):
        alpha_time_words = update_alpha_time_word(
            alpha_time_words, steps_dict["default_"], i
        )
    for key, item in steps_dict.items():
        if key != "default_":
            inds = [
                get_word_inds(prompts[i], key, tokenizer)
                for i in range(1, len(prompts))
            ]
            for i, ind in enumerate(inds):
                if len(ind) > 0:
                    alpha_time_words = update_alpha_time_word(
                        alpha_time_words, item, i, ind
                    )
    alpha_time_words = alpha_time_words.reshape(
        num_steps + 1, len(prompts) - 1, 1, 1, max_num_words
    )
    return alpha_time_words
