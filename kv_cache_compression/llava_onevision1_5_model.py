"""LLaVA-OneVision-1.5 KV-cache compression hooks.

This module contains remote-code-specific patches for
`LLaVA-OneVision-1.5-8B-Instruct`. Its text backbone and vision tower use
custom class families that are not compatible with the existing Qwen2 /
Qwen2.5-VL patch paths.
"""

from __future__ import annotations

import math
import sys
from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast

from .qwen2vl_model import get_retained_visual_positions_dart
from .siglip_model import complement_idx_prumerge_plus, outlier_dectection_prumerge_plus

TEXT_ATTENTION_FAMILY = {
    "LLaVAOneVision1_5_Attention",
    "LLaVAOneVision1_5_FlashAttention2",
    "LLaVAOneVision1_5_SdpaAttention",
}
VISION_ATTENTION_FAMILY = {
    "RiceAttention",
    "RiceFlashAttention2",
    "RiceSdpaAttention",
}
TEXT_FLASH_CLASS = "LLaVAOneVision1_5_FlashAttention2"
VISION_FLASH_CLASS = "RiceFlashAttention2"
TEXT_MODEL_CLASS = "LLaVAOneVision1_5_TextModel"
VISION_MODEL_CLASS = "RiceTransformerPretrainedModel"
GENERATION_MODEL_CLASS = "LLaVAOneVision1_5_ForConditionalGeneration"


def _get_remote_module_from_instance(instance: object) -> Any:
    module_name = instance.__class__.__module__
    module = sys.modules.get(module_name)
    if module is None:
        raise RuntimeError(
            f"Remote module '{module_name}' is not loaded for {instance.__class__.__name__}."
        )
    return module


def safe_lookup_class(module: Any, class_name: str) -> type:
    """Look up a required remote-code class and fail clearly when absent."""
    cls = getattr(module, class_name, None)
    if cls is None:
        raise RuntimeError(
            f"Required remote-code class '{class_name}' was not found in "
            f"module '{module.__name__}'."
        )
    return cls


def is_llava_onevision1_5_remote_model(model: object) -> bool:
    """Return whether the loaded model is the remote-code LLaVA-OV-1.5 model."""
    class_name = model.__class__.__name__
    module_name = model.__class__.__module__.lower()
    if class_name == GENERATION_MODEL_CLASS and "llavaonevision1_5" in module_name:
        return True

    family = get_llava_onevision1_5_remote_family_counts(model)
    return family["text_total"] > 0 and family["vision_total"] > 0


def get_llava_onevision1_5_remote_family_counts(model: object) -> dict[str, int]:
    """Count remote-code class families for routing and logging."""
    counts = {
        "text_attention": 0,
        "text_flash_attention": 0,
        "text_sdpa_attention": 0,
        "text_eager_attention": 0,
        "text_model": 0,
        "vision_attention": 0,
        "vision_flash_attention": 0,
        "vision_sdpa_attention": 0,
        "vision_eager_attention": 0,
        "vision_model": 0,
        "generation_model": 0,
    }

    for _, module in model.named_modules():
        name = module.__class__.__name__
        if name in TEXT_ATTENTION_FAMILY:
            counts["text_attention"] += 1
        if name == TEXT_FLASH_CLASS:
            counts["text_flash_attention"] += 1
        elif name == "LLaVAOneVision1_5_SdpaAttention":
            counts["text_sdpa_attention"] += 1
        elif name == "LLaVAOneVision1_5_Attention":
            counts["text_eager_attention"] += 1

        if name in VISION_ATTENTION_FAMILY:
            counts["vision_attention"] += 1
        if name == VISION_FLASH_CLASS:
            counts["vision_flash_attention"] += 1
        elif name == "RiceSdpaAttention":
            counts["vision_sdpa_attention"] += 1
        elif name == "RiceAttention":
            counts["vision_eager_attention"] += 1

        if name == TEXT_MODEL_CLASS:
            counts["text_model"] += 1
        elif name == VISION_MODEL_CLASS:
            counts["vision_model"] += 1
        elif name == GENERATION_MODEL_CLASS:
            counts["generation_model"] += 1

    counts["text_total"] = counts["text_attention"] + counts["text_model"]
    counts["vision_total"] = counts["vision_attention"] + counts["vision_model"]
    return counts


def _ensure_flash_attention_only(model: object, method: str) -> None:
    counts = get_llava_onevision1_5_remote_family_counts(model)
    text_flash = counts["text_flash_attention"]
    vision_flash = counts["vision_flash_attention"]

    if method in {"fastv", "dart"} and text_flash == 0:
        raise NotImplementedError(
            "LLaVA-OneVision-1.5 KV-cache compression currently supports only "
            "attn_implementation='flash_attention_2' for the remote text stack. "
            f"Found text families: flash={counts['text_flash_attention']}, "
            f"sdpa={counts['text_sdpa_attention']}, eager={counts['text_eager_attention']}."
        )

    if method in {"visionzip", "prumerge+"} and (text_flash == 0 or vision_flash == 0):
        raise NotImplementedError(
            "LLaVA-OneVision-1.5 KV-cache compression currently supports only "
            "attn_implementation='flash_attention_2' for the remote text and Rice "
            "vision stacks. "
            f"Found text families: flash={counts['text_flash_attention']}, "
            f"sdpa={counts['text_sdpa_attention']}, eager={counts['text_eager_attention']}; "
            f"vision families: flash={counts['vision_flash_attention']}, "
            f"sdpa={counts['vision_sdpa_attention']}, eager={counts['vision_eager_attention']}."
        )


def _build_text_image_mask(
    input_ids: torch.LongTensor,
    config: Any,
) -> torch.BoolTensor:
    image_token_id = getattr(config, "image_token_id", None)
    video_token_id = getattr(config, "video_token_id", None)
    mask = torch.ones_like(input_ids, dtype=torch.bool)
    if image_token_id is not None:
        mask &= input_ids != image_token_id
    if video_token_id is not None:
        mask &= input_ids != video_token_id
    return mask


def _propagate_text_image_mask(text_model: Any, text_image_mask: torch.BoolTensor) -> None:
    text_model.text_image_mask = text_image_mask
    for layer in text_model.layers:
        layer.self_attn.text_image_mask = text_image_mask


def _require_visual_positions(
    text_image_mask: Optional[torch.BoolTensor],
    method: str,
) -> torch.LongTensor:
    if text_image_mask is None:
        raise RuntimeError(
            f"{method} requires text_image_mask to be set before the text model runs."
        )
    visual_positions = (~text_image_mask[0]).nonzero(as_tuple=True)[0]
    if visual_positions.numel() == 0:
        raise RuntimeError(
            f"{method} requires image or video tokens in the prompt; none were found."
        )
    return visual_positions


def _slice_prefill_attention_mask(
    attention_mask: Optional[torch.Tensor],
    keep_indices: torch.LongTensor,
) -> Optional[torch.Tensor]:
    if attention_mask is None:
        return None
    if attention_mask.dim() == 2:
        return attention_mask[:, keep_indices]
    if attention_mask.dim() == 4:
        return attention_mask[:, :, keep_indices, :][:, :, :, keep_indices]
    raise RuntimeError(
        f"Unsupported attention_mask rank {attention_mask.dim()} for sequence slicing."
    )


def _slice_prefill_position_ids(
    position_ids: Optional[torch.Tensor],
    keep_indices: torch.LongTensor,
) -> Optional[torch.Tensor]:
    if position_ids is None:
        return None
    if position_ids.dim() == 3:
        return position_ids[:, :, keep_indices]
    if position_ids.dim() == 2:
        return position_ids[:, keep_indices]
    raise RuntimeError(
        f"Unsupported position_ids rank {position_ids.dim()} for sequence slicing."
    )


def _slice_cache_position(
    cache_position: Optional[torch.LongTensor],
    keep_indices: torch.LongTensor,
) -> Optional[torch.LongTensor]:
    if cache_position is None:
        return None
    if cache_position.dim() != 1:
        raise RuntimeError(
            f"Unsupported cache_position rank {cache_position.dim()} for slicing."
        )
    if cache_position.shape[0] == keep_indices.shape[0]:
        return cache_position
    return cache_position[keep_indices]


def _finish_causal_lm_forward(
    model: Any,
    outputs: Any,
    labels: Optional[torch.LongTensor],
    return_dict: bool,
) -> Any:
    hidden_states = outputs[0]
    logits = model.lm_head(hidden_states)

    loss = None
    if labels is not None:
        loss = model.loss_function(
            logits=logits,
            labels=labels,
            vocab_size=model.config.vocab_size,
        )

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    module = _get_remote_module_from_instance(model)
    output_cls = getattr(module, "LLaVAOneVision1_5_CausalLMOutputWithPast")
    return output_cls(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=outputs.rope_deltas,
    )


def _text_flash_attention_forward_impl(
    self: Any,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    position_ids: Optional[torch.LongTensor],
    past_key_value: Optional[Cache],
    output_attentions: bool,
    use_cache: bool,
    cache_position: Optional[torch.LongTensor],
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]],
    *,
    store_last_attention: bool,
    store_last_k_states: bool,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
    if self.__class__.__name__ != TEXT_FLASH_CLASS:
        raise NotImplementedError(
            "LLaVA-OneVision-1.5 KV-cache compression currently supports only "
            "attn_implementation='flash_attention_2'."
        )

    module = _get_remote_module_from_instance(self)
    apply_rotary_pos_emb = getattr(module, "apply_rotary_pos_emb")
    repeat_kv = getattr(module, "repeat_kv")
    flash_attention_forward = getattr(module, "_flash_attention_forward")

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)
    q_len = input_shape[1]

    if self.layer_idx == self.target_layer_idx - 1:
        if store_last_attention:
            self.last_attention = None
        if store_last_k_states:
            self.last_k_states = None

    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(
            key_states,
            value_states,
            self.layer_idx,
            cache_kwargs,
        )

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    if self.layer_idx == self.target_layer_idx - 1:
        if store_last_attention and q_len > 1:
            attn_scores = torch.matmul(
                query_states.float(),
                key_states.transpose(2, 3).float(),
            ) * self.scaling
            if attention_mask is not None:
                causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
                attn_scores = attn_scores + causal_mask
            else:
                causal = torch.triu(
                    torch.ones((q_len, q_len), device=query_states.device, dtype=torch.bool),
                    diagonal=1,
                )
                attn_scores = attn_scores.masked_fill(
                    causal.unsqueeze(0).unsqueeze(0),
                    float("-inf"),
                )
            self.last_attention = nn.functional.softmax(
                attn_scores,
                dim=-1,
                dtype=torch.float32,
            )
        if store_last_k_states:
            self.last_k_states = key_states

    dropout_rate = 0.0 if not self.training else self.attention_dropout

    input_dtype = query_states.dtype
    if input_dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        elif hasattr(self.config, "_pre_quantization_dtype"):
            target_dtype = self.config._pre_quantization_dtype
        else:
            target_dtype = self.q_proj.weight.dtype
        query_states = query_states.to(target_dtype)
        key_states = key_states.to(target_dtype)
        value_states = value_states.to(target_dtype)

    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    if (
        self.config.use_sliding_window
        and getattr(self.config, "sliding_window", None) is not None
        and self.layer_idx >= self.config.max_window_layers
    ):
        sliding_window = self.config.sliding_window
    else:
        sliding_window = None

    attn_output = flash_attention_forward(
        query_states,
        key_states,
        value_states,
        attention_mask,
        q_len,
        dropout=dropout_rate,
        sliding_window=sliding_window,
        is_causal=self.is_causal,
        use_top_left_mask=self._flash_attn_uses_top_left_mask,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    attn_weights = None if not output_attentions else None
    return attn_output, attn_weights, past_key_value


def llava_onevision1_5_text_attention_forward_fastv(
    self: Any,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
    """FastV attention hook for the remote LLaVA-OV-1.5 text stack."""
    return _text_flash_attention_forward_impl(
        self,
        hidden_states,
        attention_mask,
        position_ids,
        past_key_value,
        output_attentions,
        use_cache,
        cache_position,
        position_embeddings,
        store_last_attention=True,
        store_last_k_states=False,
    )


def llava_onevision1_5_text_attention_forward_dart(
    self: Any,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
    """DART attention hook for the remote LLaVA-OV-1.5 text stack."""
    return _text_flash_attention_forward_impl(
        self,
        hidden_states,
        attention_mask,
        position_ids,
        past_key_value,
        output_attentions,
        use_cache,
        cache_position,
        position_embeddings,
        store_last_attention=False,
        store_last_k_states=True,
    )


def _fastv_keep_indices(
    text_model: Any,
    hidden_states: torch.Tensor,
) -> torch.LongTensor:
    text_image_mask = getattr(text_model, "text_image_mask", None)
    visual_positions = _require_visual_positions(text_image_mask, "FastV")
    last_attention = text_model.layers[text_model.target_layer_idx - 1].self_attn.last_attention
    if last_attention is None:
        raise RuntimeError("FastV expected last_attention at layer K-1, but it was missing.")

    if hidden_states.shape[0] != 1:
        raise RuntimeError("FastV currently supports only batch_size=1 for LLaVA-OV-1.5.")

    if getattr(text_model, "origin", False):
        visual_scores = last_attention.mean(dim=1)[0][-1][visual_positions]
    else:
        visual_scores = last_attention.mean(dim=1)[0][:, visual_positions].mean(dim=0)

    keep_visual = max(1, int(visual_positions.numel() * text_model.budgets))
    selected_visual = visual_positions[visual_scores.topk(keep_visual).indices]
    keep_indices = torch.cat(
        [
            torch.where(text_image_mask[0])[0],
            selected_visual,
        ]
    ).sort().values
    return keep_indices


def _dart_keep_indices(
    text_model: Any,
    hidden_states: torch.Tensor,
) -> torch.LongTensor:
    text_image_mask = getattr(text_model, "text_image_mask", None)
    visual_positions = _require_visual_positions(text_image_mask, "DART")
    last_k_states = text_model.layers[text_model.target_layer_idx - 1].self_attn.last_k_states
    if last_k_states is None:
        raise RuntimeError("DART expected last_k_states at layer K-1, but they were missing.")
    if hidden_states.shape[0] != 1:
        raise RuntimeError("DART currently supports only batch_size=1 for LLaVA-OV-1.5.")

    last_layer_state = text_model.norm(hidden_states)
    retained_visual = get_retained_visual_positions_dart(
        last_layer_state=last_layer_state,
        k_states=last_k_states,
        visual_positions=visual_positions,
        budgets=text_model.budgets,
        pivot_image_token=text_model.pivot_image_token,
        pivot_text_token=text_model.pivot_text_token,
    )
    keep_indices = torch.cat(
        [
            torch.where(text_image_mask[0])[0],
            retained_visual.to(hidden_states.device),
        ]
    ).sort().values
    return keep_indices


def _text_model_forward_impl(
    self: Any,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    *,
    method: str,
) -> Union[Tuple[Any, ...], BaseModelOutputWithPast]:
    output_attentions = (
        output_attentions if output_attentions is not None else self.config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if self.gradient_checkpointing and self.training:
        if use_cache:
            use_cache = False

    if use_cache and past_key_values is None and not torch.jit.is_tracing():
        past_key_values = DynamicCache()

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    batch_size, seq_length = inputs_embeds.shape[:2]

    if cache_position is None:
        past_seen_tokens = (
            past_key_values.get_seq_length() if past_key_values is not None else 0
        )
        cache_position = torch.arange(
            past_seen_tokens,
            past_seen_tokens + seq_length,
            device=inputs_embeds.device,
        )

    if position_ids is None:
        position_ids = cache_position.view(1, 1, -1).expand(3, batch_size, -1)

    causal_mask = self._update_causal_mask(
        attention_mask,
        inputs_embeds,
        cache_position,
        past_key_values,
        output_attentions,
    )

    hidden_states = inputs_embeds
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = None

    for decoder_layer in self.layers:
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if decoder_layer.self_attn.layer_idx == self.target_layer_idx and seq_length > 1:
            if method == "fastv":
                keep_indices = _fastv_keep_indices(self, hidden_states)
            elif method == "dart":
                keep_indices = _dart_keep_indices(self, hidden_states)
            else:
                raise RuntimeError(f"Unsupported text pruning method '{method}'.")

            hidden_states = hidden_states[:, keep_indices, :]
            seq_length = hidden_states.shape[1]
            causal_mask = _slice_prefill_attention_mask(causal_mask, keep_indices)
            position_ids = _slice_prefill_position_ids(position_ids, keep_indices)
            cache_position = _slice_cache_position(cache_position, keep_indices)
            text_image_mask = getattr(self, "text_image_mask", None)
            if text_image_mask is not None:
                text_image_mask = text_image_mask[:, keep_indices]
                _propagate_text_image_mask(self, text_image_mask)
            position_embeddings = self.rotary_emb(hidden_states, position_ids)

        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_value=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )

        hidden_states = layer_outputs[0]
        if use_cache:
            next_decoder_cache = layer_outputs[2 if output_attentions else 1]
        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = next_decoder_cache if use_cache else None
    if not return_dict:
        return tuple(
            value
            for value in [hidden_states, next_cache, all_hidden_states, all_self_attns]
            if value is not None
        )
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )


def llava_onevision1_5_text_model_forward_fastv(
    self: Any,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
) -> Union[Tuple[Any, ...], BaseModelOutputWithPast]:
    """FastV text-model hook for the remote LLaVA-OV-1.5 text stack."""
    return _text_model_forward_impl(
        self,
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
        method="fastv",
    )


def llava_onevision1_5_text_model_forward_dart(
    self: Any,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
) -> Union[Tuple[Any, ...], BaseModelOutputWithPast]:
    """DART text-model hook for the remote LLaVA-OV-1.5 text stack."""
    return _text_model_forward_impl(
        self,
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
        method="dart",
    )


def _rice_flash_attention_forward_impl(
    self: Any,
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    rotary_pos_emb: Optional[torch.Tensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    *,
    score_attr: str,
) -> torch.Tensor:
    if self.__class__.__name__ != VISION_FLASH_CLASS:
        raise NotImplementedError(
            "LLaVA-OneVision-1.5 vision compression currently supports only "
            "attn_implementation='flash_attention_2'."
        )

    module = _get_remote_module_from_instance(self)
    apply_rotary_pos_emb_vision = getattr(module, "apply_rotary_pos_emb_vision")
    flash_attn_varlen_func = getattr(module, "flash_attn_varlen_func")

    seq_length = hidden_states.shape[0]
    query_states, key_states, value_states = (
        self.qkv(hidden_states)
        .reshape(seq_length, 3, self.num_heads, -1)
        .permute(1, 0, 2, 3)
        .unbind(0)
    )

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb_vision(
        query_states,
        key_states,
        cos,
        sin,
    )

    if self.layer_idx == self.target_layer_idx:
        self.metric = key_states.reshape(seq_length, -1)
        importance = torch.zeros(seq_length, dtype=torch.float32, device=hidden_states.device)
        for idx in range(1, len(cu_seqlens)):
            start = cu_seqlens[idx - 1].item()
            end = cu_seqlens[idx].item()
            if end - start <= 1:
                continue
            seg_q = query_states[start + 1 : end].transpose(0, 1)
            seg_k = key_states[start + 1 : end].transpose(0, 1)
            seg_attn = torch.matmul(
                seg_q.float(),
                seg_k.transpose(1, 2).float(),
            ) / math.sqrt(query_states.shape[-1])
            seg_attn = nn.functional.softmax(seg_attn, dim=-1, dtype=torch.float32)
            importance[start + 1 : end] = seg_attn.mean(dim=0).mean(dim=0)
        setattr(self, score_attr, importance)

    max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
    attn_output = flash_attn_varlen_func(
        query_states,
        key_states,
        value_states,
        cu_seqlens,
        cu_seqlens,
        max_seqlen,
        max_seqlen,
    ).reshape(seq_length, -1)
    attn_output = self.proj(attn_output)
    return attn_output


def llava_onevision1_5_rice_attention_forward_visionzip(
    self: Any,
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    rotary_pos_emb: Optional[torch.Tensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> torch.Tensor:
    """VisionZip Rice attention hook for the remote Rice vision tower."""
    return _rice_flash_attention_forward_impl(
        self,
        hidden_states,
        cu_seqlens,
        rotary_pos_emb,
        position_embeddings,
        score_attr="attn_weights",
    )


def llava_onevision1_5_rice_attention_forward_prumerge_plus(
    self: Any,
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    rotary_pos_emb: Optional[torch.Tensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> torch.Tensor:
    """PruMerge+ Rice attention hook for the remote Rice vision tower."""
    return _rice_flash_attention_forward_impl(
        self,
        hidden_states,
        cu_seqlens,
        rotary_pos_emb,
        position_embeddings,
        score_attr="attn_weights_prumerge_plus",
    )


def llava_onevision1_5_rice_block_forward_visionzip(
    self: Any,
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    rotary_pos_emb: Optional[torch.Tensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> torch.Tensor:
    """Native Rice block forward preserved for patched VisionZip execution."""
    hidden_states = hidden_states + self.attn(
        self.norm1(hidden_states),
        cu_seqlens=cu_seqlens,
        rotary_pos_emb=rotary_pos_emb,
        position_embeddings=position_embeddings,
    )
    hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
    return hidden_states


def llava_onevision1_5_rice_block_forward_prumerge_plus(
    self: Any,
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    rotary_pos_emb: Optional[torch.Tensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> torch.Tensor:
    """Native Rice block forward preserved for patched PruMerge+ execution."""
    hidden_states = hidden_states + self.attn(
        self.norm1(hidden_states),
        cu_seqlens=cu_seqlens,
        rotary_pos_emb=rotary_pos_emb,
        position_embeddings=position_embeddings,
    )
    hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
    return hidden_states


def _remove_segment_cls_tokens(
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> torch.Tensor:
    total_patches = int(cu_seqlens[-1].item())
    hidden_dim = hidden_states.shape[-1]
    patch_states = hidden_states.new_empty((total_patches, hidden_dim))
    for idx in range(1, len(cu_seqlens)):
        start = cu_seqlens[idx - 1].item()
        end = cu_seqlens[idx].item()
        patch_states[start:end] = hidden_states[start + 1 : end + 1]
    return patch_states


def _remove_segment_cls_scores(
    scores: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> torch.Tensor:
    total_patches = int(cu_seqlens[-1].item())
    patch_scores = scores.new_empty((total_patches,))
    for idx in range(1, len(cu_seqlens)):
        start = cu_seqlens[idx - 1].item()
        end = cu_seqlens[idx].item()
        patch_scores[start:end] = scores[start + 1 : end + 1]
    return patch_scores


def _compress_post_merger_visionzip(
    image_features: torch.Tensor,
    attention_scores: torch.Tensor,
    metric: torch.Tensor,
    budgets: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    total_token_num = int(image_features.shape[0])
    dominant_num = min(
        total_token_num,
        max(1, int(total_token_num * budgets * 5.4 / 6.4)),
    )
    contextual_num = min(
        max(0, total_token_num - dominant_num),
        max(1, int(total_token_num * budgets * 1.0 / 6.4)),
    )

    dominant_indices = attention_scores.topk(dominant_num, dim=0).indices.sort().values
    if contextual_num == 0 or dominant_num == total_token_num:
        return image_features[dominant_indices], dominant_indices

    mask = torch.ones(total_token_num, dtype=torch.bool, device=image_features.device)
    mask.scatter_(0, dominant_indices, False)
    filtered_indices = torch.where(mask)[0]
    dominant_tokens = image_features[dominant_indices]

    metric_filtered = metric[mask]
    features_filtered = image_features[mask]
    metric_normalized = metric_filtered / metric_filtered.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    step = max(1, metric_normalized.shape[0] // contextual_num)
    target_local_indices = torch.arange(
        0,
        metric_normalized.shape[0],
        step,
        device=image_features.device,
    )[:contextual_num]
    contextual_indices = filtered_indices[target_local_indices]
    target_tokens = metric_normalized[target_local_indices]

    source_mask = ~torch.isin(
        torch.arange(metric_normalized.shape[0], device=image_features.device),
        target_local_indices,
    )
    tokens_to_merge = metric_normalized[source_mask]
    if tokens_to_merge.numel() == 0:
        contextual_tokens = features_filtered[target_local_indices]
    else:
        similarity = torch.matmul(tokens_to_merge.float(), target_tokens.transpose(0, 1).float())
        assign_one_hot = torch.zeros(
            tokens_to_merge.shape[0],
            contextual_num,
            dtype=features_filtered.dtype,
            device=image_features.device,
        )
        assign_one_hot.scatter_(1, similarity.argmax(dim=1, keepdim=True), 1)
        counts = assign_one_hot.sum(dim=0).clamp(min=1).unsqueeze(-1)
        hidden_to_merge = features_filtered[source_mask]
        aggregated_hidden = (
            torch.matmul(assign_one_hot.transpose(0, 1).float(), hidden_to_merge.float()) / counts
        ).to(features_filtered.dtype)
        contextual_tokens = features_filtered[target_local_indices] + aggregated_hidden

    all_keep_indices = torch.cat([dominant_indices, contextual_indices]).sort().values
    compressed = torch.zeros(
        (all_keep_indices.shape[0], image_features.shape[1]),
        dtype=image_features.dtype,
        device=image_features.device,
    )
    dominant_mask = torch.isin(all_keep_indices, dominant_indices)
    compressed[dominant_mask] = dominant_tokens
    compressed[~dominant_mask] = contextual_tokens
    return compressed, all_keep_indices


def _compress_post_merger_prumerge_plus(
    image_features: torch.Tensor,
    attention_scores: torch.Tensor,
    metric: torch.Tensor,
    budgets: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    image_features_b = image_features.unsqueeze(0)
    metric_b = metric.unsqueeze(0)
    cls_attn = attention_scores.unsqueeze(0)
    batch_size, total_tokens, channel_dim = image_features_b.shape
    key_dim = metric_b.shape[-1]

    reduction_ratio = outlier_dectection_prumerge_plus(cls_attn)
    budgets_token = max(int(budgets * total_tokens), 1)
    iqr_token = max(int(total_tokens * reduction_ratio), 1)

    if budgets_token > iqr_token:
        _, iqr_idx = torch.topk(cls_attn, iqr_token, dim=1, largest=True)
        idx = torch.zeros(
            (1, budgets_token),
            dtype=iqr_idx.dtype,
            device=image_features.device,
        )
        remaining = budgets_token - iqr_token
        step_length = max(1, int(total_tokens / budgets_token))
        arithmetic_seq = torch.arange(0, total_tokens, step_length, device=image_features.device)
        used = set(iqr_idx[0].tolist())
        filtered = torch.tensor(
            [value for value in arithmetic_seq.tolist() if value not in used],
            device=image_features.device,
            dtype=iqr_idx.dtype,
        )
        if filtered.numel() > remaining:
            filtered = filtered[:remaining]
        elif filtered.numel() < remaining:
            extra = torch.tensor(
                [value for value in range(total_tokens) if value not in used],
                device=image_features.device,
                dtype=iqr_idx.dtype,
            )
            extra = extra[~torch.isin(extra, filtered)]
            if extra.numel() > 0:
                filtered = torch.cat([filtered, extra[: remaining - filtered.numel()]])
        idx[0] = torch.cat([iqr_idx[0], filtered])[:budgets_token]
    else:
        _, idx = torch.topk(cls_attn, budgets_token, dim=1, largest=True)

    index_features = idx.unsqueeze(-1).expand(-1, -1, channel_dim)
    index_k = idx.unsqueeze(-1).expand(-1, -1, key_dim)
    kept_features = torch.gather(image_features_b, dim=1, index=index_features)
    kept_keys = torch.gather(metric_b, dim=1, index=index_k)
    kept_attn = torch.gather(cls_attn, dim=1, index=idx)

    compl = complement_idx_prumerge_plus(idx, total_tokens)
    dropped_features = torch.gather(
        image_features_b,
        dim=1,
        index=compl.unsqueeze(-1).expand(-1, -1, channel_dim),
    )
    dropped_keys = torch.gather(
        metric_b,
        dim=1,
        index=compl.unsqueeze(-1).expand(-1, -1, key_dim),
    )
    dropped_attn = torch.gather(cls_attn, dim=1, index=compl)

    kept_keys_norm = nn.functional.normalize(kept_keys, p=2, dim=-1)
    dropped_keys_norm = nn.functional.normalize(dropped_keys, p=2, dim=-1)
    updated_features = torch.zeros_like(kept_features)

    for batch_idx in range(batch_size):
        for keep_idx in range(kept_features.shape[1]):
            key_here = kept_keys_norm[batch_idx, keep_idx, :].unsqueeze(0).unsqueeze(0)
            before_keys = kept_keys_norm[batch_idx, :keep_idx, :].unsqueeze(0)
            after_keys = kept_keys_norm[batch_idx, keep_idx + 1 :, :].unsqueeze(0)
            before_features = kept_features[batch_idx, :keep_idx, :].unsqueeze(0)
            after_features = kept_features[batch_idx, keep_idx + 1 :, :].unsqueeze(0)
            rest_features = torch.cat(
                [before_features, after_features, dropped_features[batch_idx].unsqueeze(0)],
                dim=1,
            )
            before_attn = kept_attn[batch_idx, :keep_idx].unsqueeze(0)
            after_attn = kept_attn[batch_idx, keep_idx + 1 :].unsqueeze(0)
            rest_attn = torch.cat(
                [before_attn, after_attn, dropped_attn[batch_idx].unsqueeze(0)],
                dim=1,
            )
            rest_keys = torch.cat(
                [before_keys, after_keys, dropped_keys_norm[batch_idx].unsqueeze(0)],
                dim=1,
            )
            cosine_similarity = torch.bmm(key_here, rest_keys.transpose(1, 2))
            cluster_size = max(min(32, cosine_similarity.shape[2]), 1)
            _, cluster_idx = torch.topk(
                cosine_similarity,
                k=cluster_size,
                dim=2,
                largest=True,
            )
            cluster_tokens = rest_features[:, cluster_idx.squeeze(), :]
            weights = rest_attn[:, cluster_idx.squeeze()].unsqueeze(-1)
            weighted_avg = torch.sum(cluster_tokens * weights, dim=1)
            updated_features[batch_idx, keep_idx, :] = (
                kept_features[batch_idx, keep_idx, :] + weighted_avg.squeeze(0)
            )

    keep_indices = idx.squeeze(0).sort().values
    return updated_features.squeeze(0).to(image_features.dtype), keep_indices


def _rice_vision_tower_forward_impl(
    self: Any,
    hidden_states: torch.Tensor,
    grid_thw: torch.Tensor,
    *,
    method: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    hidden_states = self.patch_embed(hidden_states)
    rotary_pos_emb = self.rot_pos_emb(grid_thw)
    patch_count = hidden_states.shape[0]

    cu_seqlens = torch.repeat_interleave(
        grid_thw[:, 1] * grid_thw[:, 2],
        grid_thw[:, 0],
    ).cumsum(
        dim=0,
        dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
    )
    cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
    cu_long = cu_seqlens.to(torch.long)
    num_segments = cu_long.numel() - 1

    cls_token = self.class_embedding.to(hidden_states.dtype).unsqueeze(0)
    total_patches = int(cu_long[-1].item())
    hidden_dim = hidden_states.size(-1)
    total_with_cls = total_patches + num_segments
    hidden_with_cls = hidden_states.new_empty((total_with_cls, hidden_dim))
    rotary_with_cls = rotary_pos_emb.new_empty((total_with_cls, rotary_pos_emb.shape[-1]))

    write_ptr = 0
    new_cu = [0]
    for idx in range(1, num_segments + 1):
        start = cu_long[idx - 1].item()
        end = cu_long[idx].item()
        seg_len = end - start
        hidden_with_cls[write_ptr] = cls_token
        rotary_with_cls[write_ptr] = self.class_pos_emb
        hidden_with_cls[write_ptr + 1 : write_ptr + 1 + seg_len] = hidden_states[start:end]
        rotary_with_cls[write_ptr + 1 : write_ptr + 1 + seg_len] = rotary_pos_emb[start:end]
        write_ptr += 1 + seg_len
        new_cu.append(write_ptr)

    hidden_states = self.pre_layernorm(hidden_with_cls)
    cu_seqlens = torch.tensor(new_cu, device=hidden_states.device, dtype=torch.int32)
    emb = torch.cat((rotary_with_cls, rotary_with_cls), dim=-1)
    position_embeddings = (emb.cos(), emb.sin())

    target_block_idx = max(0, len(self.blocks) - 2)
    for idx, block in enumerate(self.blocks):
        block.attn.layer_idx = idx
        block.attn.target_layer_idx = target_block_idx
        hidden_states = block(
            hidden_states,
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
        )

    patch_states = _remove_segment_cls_tokens(hidden_states, cu_long)
    image_features = self.merger(patch_states)

    target_attn = self.blocks[target_block_idx].attn
    if method == "visionzip":
        attention_scores_patch = getattr(target_attn, "attn_weights", None)
    elif method == "prumerge+":
        attention_scores_patch = getattr(target_attn, "attn_weights_prumerge_plus", None)
    else:
        raise RuntimeError(f"Unsupported Rice compression method '{method}'.")
    metric_with_cls = getattr(target_attn, "metric", None)

    if attention_scores_patch is None or metric_with_cls is None:
        raise RuntimeError(
            f"Rice {method} scoring tensors were not captured at target block "
            f"{target_block_idx}."
        )

    metric_patch = _remove_segment_cls_tokens(metric_with_cls, cu_long)
    target_attn.attn_weights = None
    target_attn.attn_weights_prumerge_plus = None
    target_attn.metric = None

    spatial_merge_unit = self.spatial_merge_size ** 2
    if patch_count % spatial_merge_unit != 0:
        raise RuntimeError(
            "Rice post-merger compression expected the patch count to be divisible "
            f"by {spatial_merge_unit}, but got {patch_count}."
        )

    attention_scores = _remove_segment_cls_scores(attention_scores_patch, cu_long)
    attention_scores = attention_scores.view(-1, spatial_merge_unit).mean(dim=1)
    metric = metric_patch.view(-1, spatial_merge_unit, metric_patch.shape[-1]).mean(dim=1)

    if method == "visionzip":
        return _compress_post_merger_visionzip(
            image_features,
            attention_scores,
            metric,
            self.budgets,
        )
    return _compress_post_merger_prumerge_plus(
        image_features,
        attention_scores,
        metric,
        self.budgets,
    )


def llava_onevision1_5_vision_tower_forward_visionzip(
    self: Any,
    hidden_states: torch.Tensor,
    grid_thw: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """VisionZip Rice vision-tower hook for the remote LLaVA-OV-1.5 model."""
    return _rice_vision_tower_forward_impl(
        self,
        hidden_states,
        grid_thw,
        method="visionzip",
    )


def llava_onevision1_5_vision_tower_forward_prumerge_plus(
    self: Any,
    hidden_states: torch.Tensor,
    grid_thw: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """PruMerge+ Rice vision-tower hook for the remote LLaVA-OV-1.5 model."""
    return _rice_vision_tower_forward_impl(
        self,
        hidden_states,
        grid_thw,
        method="prumerge+",
    )


def _apply_compressed_visual_sequence(
    model: Any,
    input_ids: torch.LongTensor,
    inputs_embeds: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    position_ids: Optional[torch.Tensor],
    cache_position: Optional[torch.LongTensor],
    pixel_values: Optional[torch.Tensor],
    image_grid_thw: Optional[torch.LongTensor],
    pixel_values_videos: Optional[torch.Tensor],
    video_grid_thw: Optional[torch.LongTensor],
) -> tuple[
    torch.LongTensor,
    torch.Tensor,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.LongTensor],
]:
    if input_ids.shape[0] != 1:
        raise RuntimeError(
            "LLaVA-OneVision-1.5 vision compression currently supports only "
            "batch_size=1."
        )

    image_token_id = model.config.image_token_id
    video_token_id = model.config.video_token_id
    image_positions = (input_ids[0] == image_token_id).nonzero(as_tuple=True)[0]
    video_positions = (input_ids[0] == video_token_id).nonzero(as_tuple=True)[0]

    keep_parts = [torch.where((input_ids[0] != image_token_id) & (input_ids[0] != video_token_id))[0]]
    compressed_input_ids = input_ids

    if pixel_values is not None:
        pixel_values = pixel_values.type(model.model.visual.dtype)
        image_embeds, image_keep = model.model.visual(pixel_values, grid_thw=image_grid_thw)
        if image_positions.numel() == 0:
            raise RuntimeError(
                "VisionZip/PruMerge+ received pixel_values but the prompt has no "
                "image tokens."
            )
        keep_parts.append(image_positions[image_keep])
    else:
        image_embeds = None

    if pixel_values_videos is not None:
        pixel_values_videos = pixel_values_videos.type(model.model.visual.dtype)
        video_embeds, video_keep = model.model.visual(
            pixel_values_videos,
            grid_thw=video_grid_thw,
        )
        if video_positions.numel() == 0:
            raise RuntimeError(
                "VisionZip/PruMerge+ received pixel_values_videos but the prompt has "
                "no video tokens."
            )
        keep_parts.append(video_positions[video_keep])
    else:
        video_embeds = None

    keep_indices = torch.cat(keep_parts).sort().values
    compressed_input_ids = input_ids[:, keep_indices]
    inputs_embeds = inputs_embeds[:, keep_indices, :]

    if image_embeds is not None:
        image_mask = (
            (compressed_input_ids == image_token_id)
            .unsqueeze(-1)
            .expand_as(inputs_embeds)
            .to(inputs_embeds.device)
        )
        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

    if video_embeds is not None:
        video_mask = (
            (compressed_input_ids == video_token_id)
            .unsqueeze(-1)
            .expand_as(inputs_embeds)
            .to(inputs_embeds.device)
        )
        video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

    attention_mask = _slice_prefill_attention_mask(attention_mask, keep_indices)
    position_ids = _slice_prefill_position_ids(position_ids, keep_indices)
    cache_position = _slice_cache_position(cache_position, keep_indices)

    _propagate_text_image_mask(
        model.model.language_model,
        _build_text_image_mask(compressed_input_ids, model.config),
    )
    return compressed_input_ids, inputs_embeds, attention_mask, position_ids, cache_position


def _generation_forward_vision_method_impl(
    self: Any,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    rope_deltas: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
) -> Any:
    output_attentions = (
        output_attentions if output_attentions is not None else self.config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if inputs_embeds is None:
        inputs_embeds = self.model.get_input_embeddings()(input_ids)
        if pixel_values is not None or pixel_values_videos is not None:
            (
                _,
                inputs_embeds,
                attention_mask,
                position_ids,
                cache_position,
            ) = _apply_compressed_visual_sequence(
                self,
                input_ids,
                inputs_embeds,
                attention_mask,
                position_ids,
                cache_position,
                pixel_values,
                image_grid_thw,
                pixel_values_videos,
                video_grid_thw,
            )
        elif attention_mask is not None:
            attention_mask = attention_mask.to(inputs_embeds.device)

    outputs = self.model(
        input_ids=None,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        cache_position=cache_position,
    )
    return _finish_causal_lm_forward(self, outputs, labels, return_dict)


def llava_onevision1_5_generation_forward_visionzip(
    self: Any,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    rope_deltas: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
) -> Any:
    """VisionZip generation hook for the remote LLaVA-OV-1.5 model."""
    return _generation_forward_vision_method_impl(
        self,
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        labels=labels,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        pixel_values=pixel_values,
        pixel_values_videos=pixel_values_videos,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        rope_deltas=rope_deltas,
        cache_position=cache_position,
    )


def llava_onevision1_5_generation_forward_prumerge_plus(
    self: Any,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    rope_deltas: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
) -> Any:
    """PruMerge+ generation hook for the remote LLaVA-OV-1.5 model."""
    return _generation_forward_vision_method_impl(
        self,
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        labels=labels,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        pixel_values=pixel_values,
        pixel_values_videos=pixel_values_videos,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        rope_deltas=rope_deltas,
        cache_position=cache_position,
    )


def _generation_forward_text_method_impl(
    self: Any,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    rope_deltas: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    *,
    method: str,
) -> Any:
    output_attentions = (
        output_attentions if output_attentions is not None else self.config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if inputs_embeds is None and input_ids is not None:
        text_image_mask = _build_text_image_mask(input_ids, self.config)
        if past_key_values is None and (~text_image_mask).sum().item() == 0:
            raise RuntimeError(
                f"{method} requires image or video tokens in the prompt for prefill pruning."
            )
        _propagate_text_image_mask(self.model.language_model, text_image_mask)

    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=True,
        pixel_values=pixel_values,
        pixel_values_videos=pixel_values_videos,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        cache_position=cache_position,
    )
    return _finish_causal_lm_forward(self, outputs, labels, return_dict)


def llava_onevision1_5_generation_forward_fastv(
    self: Any,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    rope_deltas: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
) -> Any:
    """FastV generation hook for the remote LLaVA-OV-1.5 model."""
    return _generation_forward_text_method_impl(
        self,
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        labels=labels,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        pixel_values=pixel_values,
        pixel_values_videos=pixel_values_videos,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        rope_deltas=rope_deltas,
        cache_position=cache_position,
        method="FastV",
    )


def llava_onevision1_5_generation_forward_dart(
    self: Any,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    rope_deltas: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
) -> Any:
    """DART generation hook for the remote LLaVA-OV-1.5 model."""
    return _generation_forward_text_method_impl(
        self,
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        labels=labels,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        pixel_values=pixel_values,
        pixel_values_videos=pixel_values_videos,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        rope_deltas=rope_deltas,
        cache_position=cache_position,
        method="DART",
    )
