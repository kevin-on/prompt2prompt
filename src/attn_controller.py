import abc
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as nnf
from transformers import CLIPTokenizer

import ptp_utils
import seq_aligner


@dataclass
class AttentionControlConfig:
    device: torch.device
    tokenizer: CLIPTokenizer
    max_num_words: int = 77
    low_resource: bool = False


class AttentionControl(abc.ABC):
    def step_callback(self, x_t: torch.Tensor) -> torch.Tensor:
        return x_t

    def between_steps(self) -> None:
        return

    @property
    def num_uncond_att_layers(self) -> int:
        return self.num_att_layers if self.config.low_resource else 0

    @abc.abstractmethod
    def forward(
        self, attn: torch.Tensor, is_cross: bool, place_in_unet: str
    ) -> torch.Tensor:
        raise NotImplementedError

    def __call__(
        self, attn: torch.Tensor, is_cross: bool, place_in_unet: str
    ) -> torch.Tensor:
        if self.cur_att_layer >= self.num_uncond_att_layers:
            if self.config.low_resource:
                attn = self.forward(attn, is_cross, place_in_unet)
            else:
                h = attn.shape[0]
                attn[h // 2 :] = self.forward(attn[h // 2 :], is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn

    def reset(self) -> None:
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self, config: AttentionControlConfig) -> None:
        self.config = config
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0


class EmptyControl(AttentionControl):
    def __init__(self, config: AttentionControlConfig) -> None:
        super().__init__(config)

    def forward(
        self, attn: torch.Tensor, is_cross: bool, place_in_unet: str
    ) -> torch.Tensor:
        return attn


class AttentionStore(AttentionControl):
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
        self.attention_store = {}

    def forward(
        self, attn: torch.Tensor, is_cross: bool, place_in_unet: str
    ) -> torch.Tensor:
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= 32**2:  # avoid memory overhead
            self.step_store[key].append(attn)
        return attn

    def between_steps(self) -> None:
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            # NOTE: Accumulating attention maps vs using the last step's attention map doesn't make a big difference
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]
        self.step_store = self.get_empty_store()

    def get_average_attention(self) -> Dict[str, List[torch.Tensor]]:
        average_attention: Dict[str, List[torch.Tensor]] = {
            key: [item / self.cur_step for item in self.attention_store[key]]
            for key in self.attention_store
        }
        return average_attention

    def reset(self) -> None:
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}


class LocalBlend:

    def __call__(
        self, x_t: torch.Tensor, attention_store: Dict[str, List[torch.Tensor]]
    ) -> torch.Tensor:
        k = 1
        maps = (
            attention_store["down_cross"][2:4] + attention_store["up_cross"][:3]
        )  # attention maps with 16x16 dimension
        maps = [
            item.reshape(
                self.alpha_layers.shape[0], -1, 1, 16, 16, self.config.max_num_words
            )  # batch_size x num_heads 1 x 16 x 16 x num_words
            for item in maps
        ]
        maps = torch.cat(maps, dim=1)
        maps = (maps * self.alpha_layers).sum(-1).mean(1)
        mask = nnf.max_pool2d(maps, (k * 2 + 1, k * 2 + 1), (1, 1), padding=(k, k))
        mask = nnf.interpolate(mask, size=(x_t.shape[2:]))
        mask = mask / mask.max(2, keepdims=True)[0].max(3, keepdims=True)[0]
        mask = mask.gt(self.threshold)
        mask = (mask[:1] + mask[1:]).float()
        x_t = x_t[:1] + mask * (x_t - x_t[:1])
        return x_t

    def __init__(
        self,
        config: AttentionControlConfig,
        prompts: List[str],
        words: List[List[str]],
        threshold: float = 0.3,
    ):
        self.config = config
        alpha_layers = torch.zeros(len(prompts), 1, 1, 1, 1, config.max_num_words)
        for i, (prompt, words_) in enumerate(zip(prompts, words)):
            if type(words_) is str:
                words_ = [words_]
            for word in words_:
                ind = ptp_utils.get_word_inds(prompt, word, config.tokenizer)
                alpha_layers[i, :, :, :, :, ind] = 1
        self.alpha_layers = alpha_layers.to(config.device)
        self.threshold = threshold


class AttentionControlEdit(AttentionStore, abc.ABC):

    def step_callback(self, x_t: torch.Tensor) -> torch.Tensor:
        if self.local_blend is not None:
            x_t = self.local_blend(x_t, self.attention_store)
        return x_t

    def replace_self_attention(
        self, attn_base: torch.Tensor, att_replace: torch.Tensor
    ) -> torch.Tensor:
        if att_replace.shape[2] <= 16**2:
            return attn_base.unsqueeze(0).expand(att_replace.shape[0], *attn_base.shape)
        else:
            return att_replace

    @abc.abstractmethod
    def replace_cross_attention(
        self, attn_base: torch.Tensor, att_replace: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError

    def forward(
        self, attn: torch.Tensor, is_cross: bool, place_in_unet: str
    ) -> torch.Tensor:
        super(AttentionControlEdit, self).forward(attn, is_cross, place_in_unet)
        if is_cross or (
            self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1]
        ):
            h = attn.shape[0] // (self.batch_size)
            attn = attn.reshape(self.batch_size, h, *attn.shape[1:])
            attn_base, attn_repalce = attn[0], attn[1:]
            if is_cross:
                alpha_words = self.cross_replace_alpha[self.cur_step]
                attn_repalce_new = (
                    self.replace_cross_attention(attn_base, attn_repalce) * alpha_words
                    + (1 - alpha_words) * attn_repalce
                )
                attn[1:] = attn_repalce_new
            else:
                attn[1:] = self.replace_self_attention(attn_base, attn_repalce)
            attn = attn.reshape(self.batch_size * h, *attn.shape[2:])
        return attn

    def __init__(
        self,
        config: AttentionControlConfig,
        prompts: List[str],
        num_steps: int,
        cross_replace_steps: Union[
            float, Tuple[float, float], Dict[str, Union[float, Tuple[float, float]]]
        ],
        self_replace_steps: Union[float, Tuple[float, float]],
        local_blend: Optional[LocalBlend],
    ):
        super().__init__(config)
        self.batch_size = len(prompts)
        self.cross_replace_alpha = ptp_utils.get_time_words_attention_alpha(
            prompts, num_steps, cross_replace_steps, config.tokenizer
        ).to(config.device)
        if type(self_replace_steps) is float:
            self_replace_steps = (0, self_replace_steps)
        else:
            assert isinstance(self_replace_steps, tuple)
        self.num_self_replace = int(num_steps * self_replace_steps[0]), int(
            num_steps * self_replace_steps[1]
        )
        self.local_blend = local_blend


class AttentionReplace(AttentionControlEdit):

    def replace_cross_attention(
        self, attn_base: torch.Tensor, att_replace: torch.Tensor
    ) -> torch.Tensor:
        return torch.einsum("hpw,bwn->bhpn", attn_base, self.mapper)

    def __init__(
        self,
        config: AttentionControlConfig,
        prompts: List[str],
        num_steps: int,
        cross_replace_steps: Union[
            float, Tuple[float, float], Dict[str, Union[float, Tuple[float, float]]]
        ],
        self_replace_steps: float,
        local_blend: Optional[LocalBlend] = None,
    ):
        super().__init__(
            config,
            prompts,
            num_steps,
            cross_replace_steps,
            self_replace_steps,
            local_blend,
        )
        self.mapper = seq_aligner.get_replacement_mapper(prompts, config.tokenizer).to(
            config.device
        )


class AttentionRefine(AttentionControlEdit):

    def replace_cross_attention(
        self, attn_base: torch.Tensor, att_replace: torch.Tensor
    ) -> torch.Tensor:
        attn_base_replace = attn_base[:, :, self.mapper].permute(2, 0, 1, 3)
        attn_replace = attn_base_replace * self.alphas + att_replace * (1 - self.alphas)
        return attn_replace

    def __init__(
        self,
        config: AttentionControlConfig,
        prompts: List[str],
        num_steps: int,
        cross_replace_steps: Union[
            float, Tuple[float, float], Dict[str, Union[float, Tuple[float, float]]]
        ],
        self_replace_steps: float,
        local_blend: Optional[LocalBlend] = None,
    ):
        super().__init__(
            config,
            prompts,
            num_steps,
            cross_replace_steps,
            self_replace_steps,
            local_blend,
        )
        self.mapper, alphas = seq_aligner.get_refinement_mapper(
            prompts, config.tokenizer
        )
        self.mapper, alphas = self.mapper.to(config.device), alphas.to(config.device)
        self.alphas = alphas.reshape(alphas.shape[0], 1, 1, alphas.shape[1])


class AttentionReweight(AttentionControlEdit):

    def replace_cross_attention(
        self, attn_base: torch.Tensor, att_replace: torch.Tensor
    ) -> torch.Tensor:
        if self.prev_controller is not None:
            attn_base = self.prev_controller.replace_cross_attention(
                attn_base, att_replace
            )
        attn_replace = attn_base[None, :, :, :] * self.equalizer[:, None, None, :]
        return attn_replace

    def __init__(
        self,
        config: AttentionControlConfig,
        prompts: List[str],
        num_steps: int,
        cross_replace_steps: Union[
            float, Tuple[float, float], Dict[str, Union[float, Tuple[float, float]]]
        ],
        self_replace_steps: float,
        equalizer: torch.Tensor,
        local_blend: Optional[LocalBlend] = None,
        controller: Optional[AttentionControlEdit] = None,
    ):
        super().__init__(
            config,
            prompts,
            num_steps,
            cross_replace_steps,
            self_replace_steps,
            local_blend,
        )
        self.equalizer = equalizer.to(config.device)
        self.prev_controller = controller


def get_equalizer(
    text: str,
    word_select: Union[int, str, Tuple[Union[int, str], ...]],
    values: Union[List[float], Tuple[float, ...], torch.Tensor],
    config: AttentionControlConfig,
) -> torch.Tensor:
    if isinstance(word_select, (int, str)):
        word_select = (word_select,)
    equalizer = torch.ones(len(values), config.max_num_words)
    values = torch.tensor(values, dtype=torch.float32)
    for word in word_select:
        inds = ptp_utils.get_word_inds(text, word, config.tokenizer)
        equalizer[:, inds] = values
    return equalizer
