from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from diffusers import DiffusionPipeline  # type: ignore
from einops import rearrange
from tqdm import tqdm

import ptp_utils
from attn_controller import AttentionControl, AttentionControlConfig, EmptyControl


class SingleStepAttentionStore(AttentionControl):
    @staticmethod
    def get_empty_store() -> Dict[str, List[torch.Tensor]]:
        return {
            "down_cross": [],
            "mid_cross": [],
            "up_cross": [],
            "down_self": [],
            "mid_self": [],
            "up_self": [],
        }

    def __init__(self, config: AttentionControlConfig) -> None:
        super().__init__(config)
        self.step_store = self.get_empty_store()
        self.attention_store: Dict[str, List[torch.Tensor]] = {}

    def forward(
        self, attn: torch.Tensor, is_cross: bool, place_in_unet: str
    ) -> torch.Tensor:
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= 32**2:  # avoid memory overhead
            self.step_store[key].append(attn)
        return attn

    def between_steps(self) -> None:
        self.attention_store = self.step_store
        self.step_store = self.get_empty_store()

    def reset(self) -> None:
        super().reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}


class AttentionControlMultiDiffusion(SingleStepAttentionStore):
    def __init__(
        self,
        config: AttentionControlConfig,
        num_steps: int,
        alpha: float,
    ):
        super().__init__(config)
        self.batch_size = 1  # TODO: support batch size > 1
        self.num_steps = num_steps
        self.alpha = alpha

    def set_reference(
        self,
        target_view: Optional[Tuple[int, int, int, int]],
        reference_attn_store: Optional[Dict[str, List[torch.Tensor]]],
        reference_view: Optional[Tuple[int, int, int, int]],
    ):
        self.target_view = target_view
        self.reference_attn_store = reference_attn_store
        self.reference_view = reference_view

    def forward(
        self, attn: torch.Tensor, is_cross: bool, place_in_unet: str
    ) -> torch.Tensor:
        # store the attention map
        attn = super().forward(attn, is_cross, place_in_unet)

        if (
            self.reference_attn_store is None
            or self.reference_view is None
            or self.target_view is None
        ):
            return attn

        # No replacement for self attention
        if not is_cross:
            return attn

        # Attention map larger than 32x32 is not stored
        if not attn.shape[1] <= 32**2:
            return attn

        # Get the corresponding reference attention map
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        idx = len(self.step_store[key]) - 1
        ref_attn = self.reference_attn_store[key][idx]

        # Compute the overlap bounding box
        attn_size = int(np.sqrt(attn.shape[1]))
        overlap_bbox = self.compute_overlap_bbox(
            self.target_view, self.reference_view, (attn_size, attn_size)
        )
        if overlap_bbox is None:
            return attn

        # Interpolate the attention map
        attn = rearrange(attn, "b (h w) k -> b k h w", h=attn_size, w=attn_size)
        ref_attn = rearrange(ref_attn, "b (h w) k -> b k h w", h=attn_size, w=attn_size)
        attn = self.interpolate_attention(attn, ref_attn, overlap_bbox, self.alpha)
        attn = rearrange(attn, "b k h w -> b (h w) k")

        return attn

    @staticmethod
    def compute_overlap_bbox(
        attn_coords: Tuple[int, int, int, int],
        ref_coords: Tuple[int, int, int, int],
        attn_shape: Tuple[int, int],
    ):
        """
        Computes the overlapping region (bounding box) in attention map coordinates.

        Parameters:
            attn_coords: Tuple (h_start, h_end, w_start, w_end) for the coordinates of attn's crop
            ref_coords: Tuple (h_start, h_end, w_start, w_end) for the coordinates of reference_attn's crop
            attn_shape: Tuple (H, W) of the attention map (e.g. (32, 32) or (16, 16))

        Returns:
            A tuple (row_start, row_end, col_start, col_end) in the attention map coordinates,
            or None if there is no overlap.
        """
        # Unpack coordinates
        attn_y0, attn_y1, attn_x0, attn_x1 = attn_coords
        ref_y0, ref_y1, ref_x0, ref_x1 = ref_coords

        # Intersection in the original image coordinates
        inter_x0 = max(attn_x0, ref_x0)
        inter_y0 = max(attn_y0, ref_y0)
        inter_x1 = min(attn_x1, ref_x1)
        inter_y1 = min(attn_y1, ref_y1)

        if inter_x0 >= inter_x1 or inter_y0 >= inter_y1:
            # No overlapping region
            return None

        # Compute scaling factor (each cell in attn represents these many original pixels)
        scale_x = (attn_x1 - attn_x0) / attn_shape[1]
        scale_y = (attn_y1 - attn_y0) / attn_shape[0]

        # Convert the original overlap coordinates to attention map coordinates
        # Note: we subtract the top-left of attn's crop so that the box is relative to attn
        row_start = int((inter_y0 - attn_y0) / scale_y)
        row_end = int(np.ceil((inter_y1 - attn_y0) / scale_y))
        col_start = int((inter_x0 - attn_x0) / scale_x)
        col_end = int(np.ceil((inter_x1 - attn_x0) / scale_x))

        return (row_start, row_end, col_start, col_end)

    @staticmethod
    def interpolate_attention(
        attn: torch.Tensor,
        reference_attn: torch.Tensor,
        overlap_bbox: Tuple[int, int, int, int],
        alpha: float,
    ) -> torch.Tensor:
        """
        Interpolates the overlapping region of two attention maps.

        Parameters:
            attn: Torch tensor representing the original attention map.
            reference_attn: Torch tensor representing the reference attention map.
                            Both maps are assumed to have the same shape.
            overlap_bbox: Tuple (y0, y1, x0, x1) defining the overlapping region.
            alpha: Interpolation weight for reference_attn (e.g. 0.0 to 1.0).

        Returns:
            A new attention map with the overlapping region blended.
            Outside the overlap, the original attn values are preserved.
        """
        new_attn = attn.clone()
        y0, y1, x0, x1 = overlap_bbox

        new_attn[..., y0:y1, x0:x1] = (1 - alpha) * attn[
            ..., y0:y1, x0:x1
        ] + alpha * reference_attn[..., y0:y1, x0:x1]
        return new_attn


def get_views(
    height: int, width: int, stride: int = 32, window_size: int = 512
) -> List[Tuple[int, int, int, int]]:
    if stride % 32 != 0:
        raise ValueError("stride must be divisible by 32")
    if (height - window_size) % stride != 0 or (width - window_size) % stride != 0:
        raise ValueError(
            f"Image dimensions ({height}x{width}) minus window_size ({window_size}) "
            f"must be divisible by stride ({stride}). "
            "Please adjust your image dimensions, window_size, or stride."
        )
    views = []
    for i in range(0, height - window_size + 1, stride):
        for j in range(0, width - window_size + 1, stride):
            views.append((i, i + window_size, j, j + window_size))
    return views


@torch.no_grad()
def generate(
    model: DiffusionPipeline,
    prompt: str,
    height: int,
    width: int,
    config: AttentionControlConfig,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    alpha: float = 1.0,
    baseline: bool = False,
    generator: Optional[torch.Generator] = None,
) -> np.ndarray:
    if baseline:
        controller = EmptyControl(config)
    else:
        controller = AttentionControlMultiDiffusion(
            config,
            num_steps=num_inference_steps,
            alpha=alpha,
        )
    ptp_utils.register_attention_control(model, controller)

    batch_size = 1  # TODO: support batch size > 1
    text_input = model.tokenizer(
        [prompt],
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
    if not config.low_resource:
        context = torch.cat(context)

    # set timesteps
    # "offset" is depreacted since diffusers 0.4.0
    # extra_set_kwargs = {"offset": 1}
    model.scheduler.set_timesteps(num_inference_steps)

    views = get_views(height, width, stride=32, window_size=512)
    print(f"MultiDiffusion views: {views}")

    latents = torch.randn(
        (batch_size, model.unet.config.in_channels, height // 8, width // 8),
        generator=generator,
    ).to(model.device)

    last_attn_store: Optional[Dict[str, List[torch.Tensor]]] = None
    last_view: Optional[Tuple[int, int, int, int]] = None
    latents_count = torch.zeros_like(latents)
    latents_sum = torch.zeros_like(latents)

    for t in tqdm(model.scheduler.timesteps):
        last_attn_store = None
        last_view = None
        latents_count.zero_()
        latents_sum.zero_()

        for h_start, h_end, w_start, w_end in views:
            # Configure the controller
            if isinstance(controller, AttentionControlMultiDiffusion):
                controller.set_reference(
                    (h_start, h_end, w_start, w_end),
                    last_attn_store,
                    last_view,
                )

            # PNDM scheduler needs to be set again
            model.scheduler.set_timesteps(num_inference_steps)

            h_start_latent = h_start // 8
            h_end_latent = h_end // 8
            w_start_latent = w_start // 8
            w_end_latent = w_end // 8

            latents_view = latents[
                ..., h_start_latent:h_end_latent, w_start_latent:w_end_latent
            ].clone()

            latents_view = ptp_utils.diffusion_step(
                model,
                controller,
                latents_view,
                context,
                t,
                guidance_scale,
                config.low_resource,
            )

            latents_count[
                ..., h_start_latent:h_end_latent, w_start_latent:w_end_latent
            ] += 1
            latents_sum[
                ..., h_start_latent:h_end_latent, w_start_latent:w_end_latent
            ] += latents_view

            if isinstance(controller, AttentionControlMultiDiffusion):
                last_attn_store = controller.attention_store
                last_view = (h_start, h_end, w_start, w_end)

        latents = torch.where(
            latents_count > 0, latents_sum / latents_count, latents_sum
        )

    images = ptp_utils.latent2image(model.vae, latents)
    return images
