"""
Qwen2.5-VL KV-cache compression implementations.
Adapted from qwen2vl_model.py for the Qwen2.5-VL architecture which differs in:
  - Vision transformer uses windowed attention + fullatt_block_indexes
  - Text model is Qwen2_5_VLTextModel (inside Qwen2_5_VLModel)
  - Position embeddings pre-computed and passed to each layer
  - Causal mask is a dict keyed by attention_type
"""
from copy import deepcopy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union

from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.utils import logging

from .siglip_model import outlier_dectection_prumerge_plus, complement_idx_prumerge_plus

logger = logging.get_logger(__name__)


def _resolve_visual_segment_from_token_mask(text_or_visual_mask: torch.Tensor) -> Tuple[int, int]:
    visual_positions = (text_or_visual_mask == False).nonzero(as_tuple=True)[0]
    assert len(visual_positions) > 0, "no visual token found"
    visual_start = int(visual_positions[0])
    visual_end = int(visual_positions[-1])
    return visual_start, visual_end - visual_start + 1


def _build_pruned_keep_indices(
    visual_start: int,
    visual_length: int,
    seq_length: int,
    visual_attention_score: torch.Tensor,
    keep_token_count: int,
    device: torch.device,
) -> torch.Tensor:
    top_attention_rank_index = visual_attention_score.topk(keep_token_count).indices + visual_start
    keep_indices = torch.cat(
        (
            torch.arange(visual_start, device=device),
            top_attention_rank_index,
            torch.arange(visual_start + visual_length, seq_length, device=device),
        )
    )
    return keep_indices.sort().values


def _gather_causal_mask_mapping(
    causal_mask_mapping: dict,
    keep_indices: torch.LongTensor,
) -> dict:
    new_mask_mapping = {}
    for key, value in causal_mask_mapping.items():
        if value is None:
            new_mask_mapping[key] = None
        else:
            new_mask_mapping[key] = value[:, :, keep_indices, :][:, :, :, keep_indices]
    return new_mask_mapping


def _get_visual_token_mask(module) -> Optional[torch.Tensor]:
    visual_token_mask = getattr(module, "visual_token_mask", None)
    if visual_token_mask is not None:
        return visual_token_mask
    return getattr(module, "text_image_mask", None)


def _set_language_model_visual_token_mask(language_model, visual_token_mask: torch.Tensor) -> None:
    language_model.visual_token_mask = visual_token_mask
    language_model.text_image_mask = visual_token_mask
    for layer in language_model.layers:
        layer.self_attn.visual_token_mask = visual_token_mask
        layer.self_attn.text_image_mask = visual_token_mask


# ---------------------------------------------------------------------------
# FastV — Vision Attention (Qwen2.5-VL text attention)
# ---------------------------------------------------------------------------

def qwen2_5vl_flash_attention_forward_fastv(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    **kwargs,
):
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
        apply_multimodal_rotary_pos_emb,
        ALL_ATTENTION_FUNCTIONS,
        eager_attention_forward,
    )
    from transformers.models.llama.modeling_llama import repeat_kv

    bsz, q_len, _ = hidden_states.size()

    if self.layer_idx == self.target_layer_idx - 1:
        self.last_attention = None

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_multimodal_rotary_pos_emb(
        query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
    )

    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

    # Store causal attention weights at the layer just before target_layer_idx
    if self.layer_idx == self.target_layer_idx - 1:
        key_states_rep = repeat_kv(key_states, self.num_key_value_groups)
        attn_weights_here = torch.matmul(
            query_states.float(), key_states_rep.transpose(2, 3).float()
        ) / math.sqrt(self.head_dim)
        attn_mask_causal = torch.triu(
            torch.ones((q_len, q_len), dtype=torch.bool, device=query_states.device),
            diagonal=1,
        )
        attn_weights_here = attn_weights_here.masked_fill(
            attn_mask_causal.unsqueeze(0).unsqueeze(0), float("-inf")
        )
        attn_weights_here = nn.functional.softmax(attn_weights_here, dim=-1, dtype=torch.float32)
        self.last_attention = attn_weights_here
        del attn_weights_here, attn_mask_causal, key_states_rep

    attention_interface = eager_attention_forward
    if self.config._attn_implementation != "eager":
        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        sliding_window=self.sliding_window,
        position_ids=position_ids,
        **kwargs,
    )

    attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights if output_attentions else None


# ---------------------------------------------------------------------------
# FastV — Text Model forward (Qwen2_5_VLTextModel)
# ---------------------------------------------------------------------------

def qwen2_5vl_text_model_forward_fastv(
    self,
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
    **kwargs,
) -> Union[Tuple, BaseModelOutputWithPast]:
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
        create_causal_mask,
        create_sliding_window_causal_mask,
    )

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if self.gradient_checkpointing and self.training:
        use_cache = False

    if use_cache and past_key_values is None and not torch.jit.is_tracing():
        past_key_values = DynamicCache(config=self.config)

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    batch_size, seq_length = inputs_embeds.shape[:2]

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + seq_length, device=inputs_embeds.device
        )

    if position_ids is None:
        position_ids = cache_position.view(1, 1, -1).expand(3, batch_size, -1)
    elif position_ids.ndim == 2:
        position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

    if position_ids.ndim == 3 and position_ids.shape[0] == 4:
        text_position_ids = position_ids[0]
        position_ids = position_ids[1:]
    else:
        text_position_ids = None

    if not isinstance(causal_mask_mapping := attention_mask, dict):
        mask_kwargs = {
            "config": self.config,
            "input_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "cache_position": cache_position,
            "past_key_values": past_key_values,
            "position_ids": text_position_ids,
        }
        causal_mask_mapping = {"full_attention": create_causal_mask(**mask_kwargs)}
        if self.has_sliding_layers:
            causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

    hidden_states = inputs_embeds
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None

    for decoder_layer in self.layers:
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # FastV pruning at target_layer_idx (prefill only)
        if decoder_layer.self_attn.layer_idx == self.target_layer_idx and seq_length > 1:
            ratio = self.budgets
            assert ratio is not None, "budgets is None"
            assert batch_size == 1, "FastV only supports batch_size=1"
            last_attention = self.layers[self.target_layer_idx - 1].self_attn.last_attention
            assert last_attention is not None, "last_attention is None at target layer"

            text_image_mask = self.text_image_mask[0]
            image_positions = (text_image_mask == False).nonzero(as_tuple=True)[0]
            assert len(image_positions) > 0, "No image tokens found; FastV requires an image"
            image_start = int(image_positions[0])
            image_end = int(image_positions[-1])
            image_length = image_end - image_start + 1

            device = hidden_states.device
            if self.origin:
                image_attention_score = last_attention.mean(dim=1)[0][-1][image_start:image_end + 1]
            else:
                image_attention_score = last_attention.mean(dim=1)[0][:, image_start:image_end + 1].mean(dim=0)

            top_k = max(1, int(image_length * ratio))
            top_attention_rank_index = image_attention_score.topk(top_k).indices + image_start
            keep_indices = torch.cat((
                torch.arange(image_start, device=device),
                top_attention_rank_index,
                torch.arange(image_length + image_start, seq_length, device=device),
            )).sort().values

            hidden_states = hidden_states[:, keep_indices, :]
            seq_length = hidden_states.shape[1]

            # Adjust causal masks
            new_mask_mapping = {}
            for k, v in causal_mask_mapping.items():
                if v is not None:
                    new_mask_mapping[k] = v[:, :, keep_indices, :][:, :, :, keep_indices]
                else:
                    new_mask_mapping[k] = None
            causal_mask_mapping = new_mask_mapping

            # Adjust position_ids and recompute rotary embeddings
            position_ids = position_ids[:, :, keep_indices]
            if text_position_ids is not None:
                text_position_ids = text_position_ids[:, keep_indices]
            cache_position = cache_position[keep_indices]
            position_embeddings = self.rotary_emb(hidden_states, position_ids)

        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=causal_mask_mapping[decoder_layer.attention_type],
            position_ids=text_position_ids,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )

        hidden_states = layer_outputs[0]
        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    if not return_dict:
        return tuple(v for v in [hidden_states, past_key_values, all_hidden_states, all_self_attns] if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )


# ---------------------------------------------------------------------------
# FastV — Generation forward (Qwen2_5_VLForConditionalGeneration)
# Sets text_image_mask on the language model before forward
# ---------------------------------------------------------------------------

def qwen2_5vl_generation_forward_fastv(
    self,
    input_ids=None,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    inputs_embeds=None,
    labels=None,
    use_cache=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict: Optional[bool] = None,
    pixel_values=None,
    pixel_values_videos=None,
    image_grid_thw=None,
    video_grid_thw=None,
    rope_deltas=None,
    cache_position=None,
    second_per_grid_ts=None,
    logits_to_keep=0,
    **kwargs,
):
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLCausalLMOutputWithPast
    from torch.nn import CrossEntropyLoss

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # Set text_image_mask for FastV pruning (prefill stage only)
    if input_ids is not None and pixel_values is not None and inputs_embeds is None:
        text_image_mask = (input_ids != self.config.image_token_id)
        self.model.language_model.text_image_mask = text_image_mask
        for layer in self.model.language_model.layers:
            layer.self_attn.text_image_mask = text_image_mask

    outputs = self.model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        pixel_values_videos=pixel_values_videos,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        second_per_grid_ts=second_per_grid_ts,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=True,
        cache_position=cache_position,
        **kwargs,
    )

    hidden_states = outputs[0]
    logits = self.lm_head(hidden_states[:, -logits_to_keep:, :] if isinstance(logits_to_keep, int) and logits_to_keep > 0 else hidden_states)

    loss = None
    if labels is not None:
        logits_for_loss = logits.float()
        shift_logits = logits_for_loss[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1).to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return Qwen2_5_VLCausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=getattr(self.model, 'rope_deltas', None),
    )


# ---------------------------------------------------------------------------
# PDrop — Attention forward for Qwen2.5-VL
# Stores causal attention on the stage just before each prune layer.
# ---------------------------------------------------------------------------

def qwen2_5vl_flash_attention_forward_pdrop(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    **kwargs,
):
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
        apply_multimodal_rotary_pos_emb,
        ALL_ATTENTION_FUNCTIONS,
        eager_attention_forward,
    )
    from transformers.models.llama.modeling_llama import repeat_kv

    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_multimodal_rotary_pos_emb(
        query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
    )

    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

    if self.layer_idx in self.pdrop_prev_layer_idxs and q_len > 1:
        key_states_rep = repeat_kv(key_states, self.num_key_value_groups)
        attn_weights_here = torch.matmul(
            query_states.float(), key_states_rep.transpose(2, 3).float()
        ) / math.sqrt(self.head_dim)
        attn_mask_causal = torch.triu(
            torch.ones((q_len, q_len), dtype=torch.bool, device=query_states.device),
            diagonal=1,
        )
        attn_weights_here = attn_weights_here.masked_fill(
            attn_mask_causal.unsqueeze(0).unsqueeze(0), float("-inf")
        )
        attn_weights_here = nn.functional.softmax(attn_weights_here, dim=-1, dtype=torch.float32)
        self.last_attention = attn_weights_here
        del attn_weights_here, attn_mask_causal, key_states_rep

    attention_interface = eager_attention_forward
    if self.config._attn_implementation != "eager":
        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        sliding_window=self.sliding_window,
        position_ids=position_ids,
        **kwargs,
    )

    attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights if output_attentions else None


# ---------------------------------------------------------------------------
# PDrop — Text Model forward for Qwen2.5-VL
# ---------------------------------------------------------------------------

def qwen2_5vl_text_model_forward_pdrop(
    self,
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
    **kwargs,
) -> Union[Tuple, BaseModelOutputWithPast]:
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
        create_causal_mask,
        create_sliding_window_causal_mask,
    )

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if self.gradient_checkpointing and self.training:
        use_cache = False

    if use_cache and past_key_values is None and not torch.jit.is_tracing():
        past_key_values = DynamicCache(config=self.config)

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    batch_size, seq_length = inputs_embeds.shape[:2]

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + seq_length, device=inputs_embeds.device
        )

    if position_ids is None:
        position_ids = cache_position.view(1, 1, -1).expand(3, batch_size, -1)
    elif position_ids.ndim == 2:
        position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

    if position_ids.ndim == 3 and position_ids.shape[0] == 4:
        text_position_ids = position_ids[0]
        position_ids = position_ids[1:]
    else:
        text_position_ids = None

    if not isinstance(causal_mask_mapping := attention_mask, dict):
        mask_kwargs = {
            "config": self.config,
            "input_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "cache_position": cache_position,
            "past_key_values": past_key_values,
            "position_ids": text_position_ids,
        }
        causal_mask_mapping = {"full_attention": create_causal_mask(**mask_kwargs)}
        if self.has_sliding_layers:
            causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

    hidden_states = inputs_embeds
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None

    pdrop_layer_to_ratio = self.pdrop_layer_to_ratio
    current_visual_token_mask = None
    original_visual_length = None
    if seq_length > 1:
        visual_token_mask = _get_visual_token_mask(self)
        assert visual_token_mask is not None, "pdrop requires a visual token mask during prefill"
        current_visual_token_mask = visual_token_mask[0]
        _, original_visual_length = _resolve_visual_segment_from_token_mask(current_visual_token_mask)

    for decoder_layer in self.layers:
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        layer_idx = decoder_layer.self_attn.layer_idx
        if layer_idx in pdrop_layer_to_ratio and seq_length > 1:
            ratio = pdrop_layer_to_ratio[layer_idx]
            assert batch_size == 1, "pdrop only supports batch_size=1"
            assert current_visual_token_mask is not None and original_visual_length is not None, (
                "pdrop requires tracked visual token positions during prefill"
            )

            last_attention = self.layers[layer_idx - 1].self_attn.last_attention
            assert last_attention is not None, (
                "last_attention is None, but it should be populated by the previous stage layer"
            )

            current_visual_start, current_visual_length = _resolve_visual_segment_from_token_mask(
                current_visual_token_mask
            )
            target_keep_tokens = max(1, int(original_visual_length * ratio))
            target_keep_tokens = min(current_visual_length, target_keep_tokens)
            visual_end = current_visual_start + current_visual_length

            if self.origin:
                visual_attention_score = last_attention.mean(dim=1)[0][-1][current_visual_start:visual_end]
            else:
                visual_attention_score = last_attention.mean(dim=1)[0][
                    :, current_visual_start:visual_end
                ].mean(dim=0)

            keep_indices = _build_pruned_keep_indices(
                current_visual_start,
                current_visual_length,
                seq_length,
                visual_attention_score,
                target_keep_tokens,
                hidden_states.device,
            )

            hidden_states = hidden_states[:, keep_indices, :]
            seq_length = hidden_states.shape[1]
            current_visual_token_mask = current_visual_token_mask[keep_indices]
            causal_mask_mapping = _gather_causal_mask_mapping(causal_mask_mapping, keep_indices)
            position_ids = position_ids[:, :, keep_indices]
            if text_position_ids is not None:
                text_position_ids = text_position_ids[:, keep_indices]
            cache_position = cache_position[keep_indices]
            position_embeddings = self.rotary_emb(hidden_states, position_ids)

        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=causal_mask_mapping[decoder_layer.attention_type],
            position_ids=text_position_ids,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )

        hidden_states = layer_outputs[0]
        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    if not return_dict:
        return tuple(v for v in [hidden_states, past_key_values, all_hidden_states, all_self_attns] if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )


# ---------------------------------------------------------------------------
# PDrop — Generation forward for Qwen2.5-VL
# Sets visual token mask on the language model before calling self.model(...)
# ---------------------------------------------------------------------------

def qwen2_5vl_generation_forward_pdrop(
    self,
    input_ids=None,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    inputs_embeds=None,
    labels=None,
    use_cache=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict: Optional[bool] = None,
    pixel_values=None,
    pixel_values_videos=None,
    image_grid_thw=None,
    video_grid_thw=None,
    rope_deltas=None,
    cache_position=None,
    second_per_grid_ts=None,
    logits_to_keep=0,
    **kwargs,
):
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLCausalLMOutputWithPast
    from torch.nn import CrossEntropyLoss

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if input_ids is not None and inputs_embeds is None:
        visual_token_mask = None
        if pixel_values is not None:
            visual_token_mask = (input_ids != self.config.image_token_id)
        elif pixel_values_videos is not None:
            visual_token_mask = (input_ids != self.config.video_token_id)

        if visual_token_mask is not None:
            _set_language_model_visual_token_mask(self.model.language_model, visual_token_mask)

    outputs = self.model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        pixel_values_videos=pixel_values_videos,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        second_per_grid_ts=second_per_grid_ts,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=True,
        cache_position=cache_position,
        **kwargs,
    )

    hidden_states = outputs[0]
    logits = self.lm_head(hidden_states[:, -logits_to_keep:, :] if isinstance(logits_to_keep, int) and logits_to_keep > 0 else hidden_states)

    loss = None
    if labels is not None:
        logits_for_loss = logits.float()
        shift_logits = logits_for_loss[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1).to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return Qwen2_5_VLCausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=getattr(self.model, 'rope_deltas', None),
    )


# ---------------------------------------------------------------------------
# VisionZip — Vision Attention (Qwen2_5_VLVisionAttention)
# Stores key states (metric) and attention weights at the target block
# ---------------------------------------------------------------------------

def qwen2_5vl_vision_attention_forward_visionzip(
    self,
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    rotary_pos_emb: Optional[torch.Tensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    **kwargs,
) -> torch.Tensor:
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
        ALL_ATTENTION_FUNCTIONS,
        eager_attention_forward,
        apply_rotary_pos_emb_vision,
    )
    target_idx = getattr(self, "target_layer_idx", -2)  # set during patching
    if self.layer_idx == target_idx:
        self.metric = None
        self.attn_weights = None

    seq_length = hidden_states.shape[0]
    query_states, key_states, value_states = (
        self.qkv(hidden_states)
        .reshape(seq_length, 3, self.num_heads, -1)
        .permute(1, 0, 2, 3)
        .unbind(0)
    )

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)

    if self.layer_idx == target_idx:
        # Store key states as the metric (in window-reordered patch space)
        self.metric = key_states.reshape(seq_length, -1)  # [seq_len, num_heads*head_dim]

        # Compute full attention for scoring
        q_t = query_states.transpose(0, 1).unsqueeze(0)   # [1, num_heads, seq_len, head_dim]
        k_t = key_states.transpose(0, 1).unsqueeze(0)
        attn_w = torch.matmul(q_t.float(), k_t.transpose(2, 3).float()) / math.sqrt(query_states.shape[-1])
        attn_mask_here = torch.full([1, seq_length, seq_length], True, dtype=torch.bool, device=hidden_states.device)
        for i in range(1, len(cu_seqlens)):
            attn_mask_here[..., cu_seqlens[i - 1]:cu_seqlens[i], cu_seqlens[i - 1]:cu_seqlens[i]] = False
        attn_w = attn_w.masked_fill(attn_mask_here, float("-inf"))
        attn_w = nn.functional.softmax(attn_w, dim=-1, dtype=torch.float32)
        self.attn_weights = attn_w.squeeze(0)  # [num_heads, seq_len, seq_len]
        del q_t, k_t, attn_w, attn_mask_here

    query_states = query_states.transpose(0, 1).unsqueeze(0)
    key_states = key_states.transpose(0, 1).unsqueeze(0)
    value_states = value_states.transpose(0, 1).unsqueeze(0)

    attention_interface = eager_attention_forward
    if self.config._attn_implementation != "eager":
        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    if self.config._attn_implementation == "flash_attention_2":
        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()
        attn_output, _ = attention_interface(
            self, query_states, key_states, value_states,
            attention_mask=None, scaling=self.scaling,
            dropout=0.0, cu_seq_lens_q=cu_seqlens, cu_seq_lens_k=cu_seqlens,
            max_length_q=max_seqlen, max_length_k=max_seqlen, is_causal=False, **kwargs,
        )
    else:
        lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        splits = [torch.split(t, lengths.tolist(), dim=2) for t in (query_states, key_states, value_states)]
        attn_outputs = [
            attention_interface(self, q, k, v, attention_mask=None, scaling=self.scaling,
                                dropout=0.0, is_causal=False, **kwargs)[0]
            for q, k, v in zip(*splits)
        ]
        attn_output = torch.cat(attn_outputs, dim=1)

    attn_output = attn_output.reshape(seq_length, -1).contiguous()
    attn_output = self.proj(attn_output)
    return attn_output


# ---------------------------------------------------------------------------
# VisionZip — Vision Block forward (Qwen2_5_VLVisionBlock) — unchanged logic
# ---------------------------------------------------------------------------

def qwen2_5vl_vision_block_forward_visionzip(
    self,
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    rotary_pos_emb: Optional[torch.Tensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    **kwargs,
) -> torch.Tensor:
    hidden_states = hidden_states + self.attn(
        self.norm1(hidden_states),
        cu_seqlens=cu_seqlens,
        position_embeddings=position_embeddings,
        **kwargs,
    )
    hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
    return hidden_states


# ---------------------------------------------------------------------------
# VisionZip — Vision Tower forward (Qwen2_5_VisionTransformerPretrainedModel)
# ---------------------------------------------------------------------------

def qwen2_5vl_vision_tower_forward_visionzip(
    self,
    hidden_states: torch.Tensor,
    grid_thw: torch.Tensor,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    hidden_states = self.patch_embed(hidden_states)
    rotary_pos_emb = self.rot_pos_emb(grid_thw)
    window_index, cu_window_seqlens = self.get_window_index(grid_thw)
    cu_window_seqlens = torch.tensor(
        cu_window_seqlens,
        device=hidden_states.device,
        dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
    )
    cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

    seq_len, _ = hidden_states.size()
    # Reorder into window groups
    hidden_states = hidden_states.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
    hidden_states = hidden_states[window_index, :, :]
    hidden_states = hidden_states.reshape(seq_len, -1)
    rotary_pos_emb = rotary_pos_emb.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
    rotary_pos_emb = rotary_pos_emb[window_index, :, :]
    rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
    emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
    position_embeddings = (emb.cos(), emb.sin())

    cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
        dim=0, dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
    )
    cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

    n_blocks = len(self.blocks)
    # Use the second-to-last fullatt block for scoring — mirrors VisionZip paper's "second-to-last layer"
    # design. The last fullatt block (block 31) sits immediately before the merger and its features may
    # be over-specialised for merger input, giving poor discriminative attention for text / OCR tokens.
    # The second-to-last fullatt block (block 23 for Qwen2.5-VL/Qwen3-VL) retains global attention but
    # still has several processing steps remaining, yielding more discriminative scores.
    fullatt_indexes = getattr(self, 'fullatt_block_indexes', None)
    if fullatt_indexes and len(fullatt_indexes) >= 2:
        target_block_idx = fullatt_indexes[-2]  # second-to-last fullatt (e.g. block 23 for Qwen2.5-VL)
    elif fullatt_indexes:
        target_block_idx = fullatt_indexes[-1]  # only one fullatt block — no choice
    else:
        target_block_idx = n_blocks - 2  # all-fullatt architecture (e.g. Qwen2-VL), use second-to-last

    for layer_num, blk in enumerate(self.blocks):
        cu_seqlens_now = cu_seqlens if layer_num in self.fullatt_block_indexes else cu_window_seqlens
        hidden_states = blk(
            hidden_states,
            cu_seqlens=cu_seqlens_now,
            position_embeddings=position_embeddings,
        )

    # Retrieve stored attention weights and metric from target block
    target_attn = self.blocks[target_block_idx].attn
    attn_weights = target_attn.attn_weights  # [num_heads, seq_len_patches, seq_len_patches]
    metric = target_attn.metric              # [seq_len_patches, num_heads*head_dim]
    target_attn.attn_weights = None
    target_attn.metric = None

    # Apply merger and reverse window ordering
    hidden_states = self.merger(hidden_states)  # [n_groups, out_hidden_size]
    n_groups = hidden_states.shape[0]
    reverse_indices = torch.argsort(window_index)
    hidden_states = hidden_states[reverse_indices, :]  # original spatial order

    # Map attention weights to group space and original order
    num_heads = attn_weights.shape[0]
    # Pool patches to groups: mean over spatial_merge_unit patches per group
    attn_sum_patches = attn_weights.mean(dim=0).mean(dim=0)  # [seq_len_patches]
    attn_sum_groups_window = attn_sum_patches.view(n_groups, self.spatial_merge_unit).mean(dim=1)
    attention_sum = attn_sum_groups_window[reverse_indices]  # [n_groups] in original order

    # Pool metric to groups and original order
    metric_groups_window = metric.view(n_groups, self.spatial_merge_unit, -1).mean(dim=1)
    metric = metric_groups_window[reverse_indices]  # [n_groups, dim]

    # VisionZip: select dominant + contextual tokens
    total_token_num = n_groups
    dominant_num = max(1, int(total_token_num * self.budgets * 5.4 / 6.4))
    contextual_num = max(1, int(total_token_num * self.budgets * 1.0 / 6.4))

    all_indices = attention_sum.topk(dominant_num, dim=0).indices
    all_indices = all_indices.sort().values

    mask = torch.ones(total_token_num, dtype=torch.bool, device=metric.device).scatter_(0, all_indices, False)

    filtered_indices = torch.where(mask)[0]
    dominant_tokens = hidden_states.masked_select(~mask.unsqueeze(-1)).view(dominant_num, hidden_states.shape[1])

    metric_filtered = metric[mask]
    hidden_states_filtered = hidden_states.masked_select(mask.unsqueeze(-1)).view(
        total_token_num - dominant_num, hidden_states.shape[1]
    )
    metric_normalized = metric_filtered / metric_filtered.norm(dim=-1, keepdim=True)

    step = max(1, metric_normalized.shape[0] // contextual_num)
    target_indices = torch.arange(0, metric_normalized.shape[0], step, device=metric_normalized.device)[:contextual_num]
    contextual_indices = filtered_indices[target_indices]
    target_tokens = metric_normalized[target_indices, :]

    tokens_to_merge = metric_normalized[
        ~torch.isin(torch.arange(metric_normalized.shape[0], device=metric_normalized.device), target_indices), :
    ]
    similarity = torch.matmul(tokens_to_merge.float(), target_tokens.transpose(0, 1).float())
    assign_one_hot = torch.zeros(
        tokens_to_merge.shape[0], contextual_num, dtype=hidden_states_filtered.dtype, device=metric_normalized.device
    )
    assign_one_hot.scatter_(1, similarity.argmax(dim=1).unsqueeze(-1), 1)
    counts = assign_one_hot.sum(dim=0).clamp(min=1).unsqueeze(-1)

    hidden_to_merge = hidden_states_filtered[
        ~torch.isin(torch.arange(hidden_states_filtered.shape[0], device=hidden_states_filtered.device), target_indices), :
    ]
    aggregated_hidden = (torch.matmul(assign_one_hot.transpose(0, 1).float(), hidden_to_merge.float()) / counts).to(torch.bfloat16)
    target_hidden = hidden_states_filtered[target_indices, :]
    contextual_tokens = target_hidden + aggregated_hidden

    all_keep_indices = torch.cat([all_indices, contextual_indices]).sort().values

    hidden_states_save = torch.zeros(
        (len(all_keep_indices), hidden_states.shape[1]),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    dominant_mask = torch.isin(all_keep_indices, all_indices)
    hidden_states_save[torch.where(dominant_mask)[0]] = dominant_tokens
    hidden_states_save[torch.where(~dominant_mask)[0]] = contextual_tokens

    return hidden_states_save, all_keep_indices


# ---------------------------------------------------------------------------
# VisionZip — Generation forward (Qwen2_5_VLForConditionalGeneration)
# ---------------------------------------------------------------------------

def qwen2_5vl_generation_forward_visionzip(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values=None,
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
    second_per_grid_ts: Optional[torch.Tensor] = None,
    logits_to_keep: int = 0,
    **kwargs,
):
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLCausalLMOutputWithPast
    from torch.nn import CrossEntropyLoss

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if inputs_embeds is None:
        inputs_embeds = self.model.get_input_embeddings()(input_ids)

        if pixel_values is not None:  # prefill stage
            pixel_values = pixel_values.type(self.model.visual.dtype)
            image_embeds, all_indices = self.model.visual(pixel_values, grid_thw=image_grid_thw)
            n_image_tokens = image_embeds.shape[0]

            total_len = input_ids.shape[-1]
            assert input_ids.shape[0] == 1, "VisionZip only supports batch_size=1"

            # Find vision start/end token positions
            vision_start_id = self.config.vision_start_token_id  # 151652
            vision_end_id = self.config.vision_end_token_id      # 151653

            position_image_begin = (input_ids[0] == vision_start_id).nonzero(as_tuple=True)[0]
            before_idx = position_image_begin[0].item() + 1
            before_img = input_ids[:, :before_idx]

            position_image_end = (input_ids[0] == vision_end_id).nonzero(as_tuple=True)[0]
            post_idx = position_image_end[-1].item()
            post_img = input_ids[:, post_idx:]

            img_tensor = torch.full(
                (input_ids.shape[0], n_image_tokens),
                self.config.image_token_id,
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            origin_input_ids = deepcopy(input_ids)
            input_ids = torch.cat((before_img, img_tensor, post_img), dim=1)

            all_indices = all_indices + before_idx
            all_indices = torch.cat((
                torch.arange(0, before_idx, device=all_indices.device),
                all_indices,
                torch.arange(post_idx, total_len, device=all_indices.device),
            ))
            inputs_embeds = inputs_embeds[:, all_indices, :]

            image_mask = (
                (input_ids == self.config.image_token_id)
                .unsqueeze(-1)
                .expand_as(inputs_embeds)
                .to(inputs_embeds.device)
            )
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            # Save all_indices for decode stage
            self.model.all_indices = all_indices

            # Save original attention_mask for get_rope_index (needed before slicing)
            _orig_attn_mask_for_rope = attention_mask
            _compressed_len = all_indices.shape[0]

            # Set text_image_mask for potential combined FastV use
            text_image_mask = (input_ids != self.config.image_token_id)
            self.model.language_model.text_image_mask = text_image_mask
            for layer in self.model.language_model.layers:
                layer.self_attn.text_image_mask = text_image_mask

        if pixel_values_videos is not None:
            pixel_values_videos = pixel_values_videos.type(self.model.visual.dtype)
            video_embeds = self.model.visual(pixel_values_videos, grid_thw=video_grid_thw)
            video_mask = (
                (input_ids == self.config.video_token_id)
                .unsqueeze(-1)
                .expand_as(inputs_embeds)
                .to(inputs_embeds.device)
            )
            video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        if attention_mask is not None:
            attention_mask = attention_mask.to(inputs_embeds.device)

        # VisionZip prefill: adjust positional information to match compressed sequence.
        # prepare_inputs_for_generation may have pre-computed position_ids for the full
        # (uncompressed) sequence. We must slice both position_ids and attention_mask
        # to match the compressed length, and correct rope_deltas for decode stage.
        if pixel_values is not None:
            if attention_mask is not None:
                attention_mask = attention_mask[:, all_indices]
            if position_ids is not None:
                position_ids = position_ids[:, :, all_indices]
            if self.model.rope_deltas is not None:
                # Correct rope_deltas: cache_position during decode = compressed KV-cache
                # size (e.g. 819), so we need rope_deltas s.t.
                #   cache_position[0] + rope_deltas_corrected = max_pos + 1
                # Original: max_pos + 1 = total_len + rope_deltas_original
                # => rope_deltas_corrected = rope_deltas_original + (total_len - compressed_len)
                self.model.rope_deltas = self.model.rope_deltas + (total_len - _compressed_len)

    # Compute position_ids from original input
    if position_ids is None and input_ids is not None and (attention_mask is None or attention_mask.ndim == 2):
        # Check if this is prefill (has pixel_values and not using KV cache)
        is_prefill = pixel_values is not None and (past_key_values is None or len(past_key_values) == 0)

        if is_prefill:
            # Prefill stage: compute rope_deltas and select by all_indices.
            # Use _orig_attn_mask_for_rope (the unsliced mask) so get_rope_index sees
            # the full original sequence; then slice the result by all_indices.
            position_ids, rope_deltas = self.model.get_rope_index(
                origin_input_ids, image_grid_thw, video_grid_thw,
                second_per_grid_ts=second_per_grid_ts, attention_mask=_orig_attn_mask_for_rope,
            )
            # Correct rope_deltas so decode positions are right after compression
            self.model.rope_deltas = rope_deltas + (total_len - _compressed_len)
            # Select positions corresponding to kept tokens
            position_ids = position_ids[:, :, all_indices]
            # Note: attention_mask was already sliced above (in the pixel_values block)
        else:
            # Decode stage: position_ids for the new token.
            # self.model.rope_deltas was corrected during prefill so that
            # cache_position[0] + rope_deltas == max_pos + 1 (correct next position).
            batch_size_here, seq_length_here, _ = inputs_embeds.shape
            position_ids = torch.arange(seq_length_here, device=inputs_embeds.device)
            position_ids = position_ids.view(1, 1, -1).expand(3, batch_size_here, -1)
            if cache_position is not None and self.model.rope_deltas is not None:
                delta = (cache_position[0] + self.model.rope_deltas).to(inputs_embeds.device)
            else:
                delta = torch.zeros((batch_size_here, seq_length_here), device=inputs_embeds.device)
            delta = delta.repeat_interleave(batch_size_here // delta.shape[0], dim=0)
            position_ids = position_ids + delta.to(position_ids.device)

    outputs = self.model(
        input_ids=None,
        pixel_values=None,  # already embedded
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=True,
        cache_position=cache_position,
    )

    hidden_states = outputs[0]
    logits = self.lm_head(hidden_states)

    loss = None
    if labels is not None:
        logits_f = logits.float()
        shift_logits = logits_f[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1).to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return Qwen2_5_VLCausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=self.model.rope_deltas,
    )


# ---------------------------------------------------------------------------
# Prumerge+ — Vision Attention (Qwen2_5_VLVisionAttention)
# ---------------------------------------------------------------------------

def qwen2_5vl_vision_attention_forward_prumerge_plus(
    self,
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    rotary_pos_emb: Optional[torch.Tensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    **kwargs,
) -> torch.Tensor:
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
        ALL_ATTENTION_FUNCTIONS,
        eager_attention_forward,
        apply_rotary_pos_emb_vision,
    )
    target_idx = getattr(self, "target_layer_idx", -2)
    if self.layer_idx == target_idx:
        self.metric = None
        self.attn_weights_prumerge_plus = None

    seq_length = hidden_states.shape[0]
    query_states, key_states, value_states = (
        self.qkv(hidden_states)
        .reshape(seq_length, 3, self.num_heads, -1)
        .permute(1, 0, 2, 3)
        .unbind(0)
    )

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)

    if self.layer_idx == target_idx:
        self.metric = key_states.reshape(seq_length, -1)
        q_t = query_states.transpose(0, 1).unsqueeze(0)
        k_t = key_states.transpose(0, 1).unsqueeze(0)
        attn_w = torch.matmul(q_t.float(), k_t.transpose(2, 3).float()) / math.sqrt(query_states.shape[-1])
        attn_mask_here = torch.full([1, seq_length, seq_length], True, dtype=torch.bool, device=hidden_states.device)
        for i in range(1, len(cu_seqlens)):
            attn_mask_here[..., cu_seqlens[i - 1]:cu_seqlens[i], cu_seqlens[i - 1]:cu_seqlens[i]] = False
        attn_w = attn_w.masked_fill(attn_mask_here, float("-inf"))
        attn_w = nn.functional.softmax(attn_w, dim=-1, dtype=torch.float32)
        self.attn_weights_prumerge_plus = attn_w.squeeze(0)
        del q_t, k_t, attn_w, attn_mask_here

    query_states = query_states.transpose(0, 1).unsqueeze(0)
    key_states = key_states.transpose(0, 1).unsqueeze(0)
    value_states = value_states.transpose(0, 1).unsqueeze(0)

    attention_interface = eager_attention_forward
    if self.config._attn_implementation != "eager":
        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    if self.config._attn_implementation == "flash_attention_2":
        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()
        attn_output, _ = attention_interface(
            self, query_states, key_states, value_states,
            attention_mask=None, scaling=self.scaling,
            dropout=0.0, cu_seq_lens_q=cu_seqlens, cu_seq_lens_k=cu_seqlens,
            max_length_q=max_seqlen, max_length_k=max_seqlen, is_causal=False, **kwargs,
        )
    else:
        lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        splits = [torch.split(t, lengths.tolist(), dim=2) for t in (query_states, key_states, value_states)]
        attn_outputs = [
            attention_interface(self, q, k, v, attention_mask=None, scaling=self.scaling,
                                dropout=0.0, is_causal=False, **kwargs)[0]
            for q, k, v in zip(*splits)
        ]
        attn_output = torch.cat(attn_outputs, dim=1)

    attn_output = attn_output.reshape(seq_length, -1).contiguous()
    attn_output = self.proj(attn_output)
    return attn_output


# ---------------------------------------------------------------------------
# Prumerge+ — Vision Block forward (Qwen2_5_VLVisionBlock)
# ---------------------------------------------------------------------------

def qwen2_5vl_vision_block_forward_prumerge_plus(
    self,
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    rotary_pos_emb: Optional[torch.Tensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    **kwargs,
) -> torch.Tensor:
    hidden_states = hidden_states + self.attn(
        self.norm1(hidden_states),
        cu_seqlens=cu_seqlens,
        position_embeddings=position_embeddings,
        **kwargs,
    )
    hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
    return hidden_states


# ---------------------------------------------------------------------------
# Prumerge+ — Vision Tower forward
# ---------------------------------------------------------------------------

def qwen2_5vl_vision_tower_forward_prumerge_plus(
    self,
    hidden_states: torch.Tensor,
    grid_thw: torch.Tensor,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    hidden_states = self.patch_embed(hidden_states)
    rotary_pos_emb = self.rot_pos_emb(grid_thw)
    window_index, cu_window_seqlens = self.get_window_index(grid_thw)
    cu_window_seqlens = torch.tensor(
        cu_window_seqlens,
        device=hidden_states.device,
        dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
    )
    cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

    seq_len, _ = hidden_states.size()
    hidden_states = hidden_states.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
    hidden_states = hidden_states[window_index, :, :]
    hidden_states = hidden_states.reshape(seq_len, -1)
    rotary_pos_emb = rotary_pos_emb.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
    rotary_pos_emb = rotary_pos_emb[window_index, :, :]
    rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
    emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
    position_embeddings = (emb.cos(), emb.sin())

    cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
        dim=0, dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
    )
    cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

    n_blocks = len(self.blocks)
    # Second-to-last fullatt block for scoring — see visionzip tower for rationale.
    fullatt_indexes = getattr(self, 'fullatt_block_indexes', None)
    if fullatt_indexes and len(fullatt_indexes) >= 2:
        target_block_idx = fullatt_indexes[-2]
    elif fullatt_indexes:
        target_block_idx = fullatt_indexes[-1]
    else:
        target_block_idx = n_blocks - 2

    for layer_num, blk in enumerate(self.blocks):
        cu_seqlens_now = cu_seqlens if layer_num in self.fullatt_block_indexes else cu_window_seqlens
        hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens_now, position_embeddings=position_embeddings)

    # Retrieve stored state
    target_attn = self.blocks[target_block_idx].attn
    attn_weights = target_attn.attn_weights_prumerge_plus
    metric = target_attn.metric
    target_attn.attn_weights_prumerge_plus = None
    target_attn.metric = None

    # Apply merger and reverse window ordering
    image_features = self.merger(hidden_states)
    n_groups = image_features.shape[0]
    reverse_indices = torch.argsort(window_index)
    image_features = image_features[reverse_indices, :]  # original order

    # Pool and un-reorder attn_weights
    num_heads = attn_weights.shape[0]
    q_len_pooled = attn_weights.shape[1] // self.spatial_merge_unit
    k_len_pooled = attn_weights.shape[2] // self.spatial_merge_unit
    attn_weights_pooled = attn_weights.view(
        num_heads, q_len_pooled, self.spatial_merge_unit, k_len_pooled, self.spatial_merge_unit
    ).mean(dim=(2, 4))  # [num_heads, q_groups, k_groups] in window order
    # Unorder: apply reverse_indices to both dims
    attn_weights_orig = attn_weights_pooled[:, reverse_indices, :][:, :, reverse_indices]

    # Pool metric
    metric_groups_window = metric.view(n_groups, self.spatial_merge_unit, -1).mean(dim=1)
    desired_layer_k = metric_groups_window[reverse_indices]  # [n_groups, dim]

    image_features_b = image_features.unsqueeze(0)  # [1, N, C]
    B, N, C = image_features_b.shape
    desired_layer_k_b = desired_layer_k.unsqueeze(0)  # [1, N, k_dim]
    k_C = desired_layer_k_b.shape[-1]

    cls_attn = torch.mean(attn_weights_orig, dim=[0, 1]).unsqueeze(0)  # [1, N]

    reduction_ratio = outlier_dectection_prumerge_plus(cls_attn)
    budgets_token = max(int(self.budgets * N), 1)
    iqr_token = max(int(N * reduction_ratio), 1)

    if budgets_token > iqr_token:
        _, iqr_idx = torch.topk(cls_attn, iqr_token, dim=1, largest=True)
        idx = torch.zeros((1, budgets_token), dtype=iqr_idx.dtype, device=self.device)
        remaining = budgets_token - iqr_token
        step_length = max(1, int(N / budgets_token))
        arithmetic_seq = torch.arange(0, N, step_length, device=self.device)
        orig_1d = iqr_idx[0].flatten()
        filtered = torch.tensor([x for x in arithmetic_seq if x not in orig_1d], device=self.device)
        if len(filtered) > remaining:
            filtered = filtered[:remaining]
        elif len(filtered) < remaining:
            avail = torch.tensor([x for x in range(N) if x not in orig_1d and x not in filtered], device=self.device)
            if len(avail) > 0:
                extra = avail[torch.randperm(len(avail))[:remaining - len(filtered)]]
                filtered = torch.cat([filtered, extra])
        idx[0] = torch.cat([iqr_idx[0], filtered])[:budgets_token]
    else:
        _, idx = torch.topk(cls_attn, budgets_token, dim=1, largest=True)

    index_features = idx.unsqueeze(-1).expand(-1, -1, C)
    index_k = idx.unsqueeze(-1).expand(-1, -1, k_C)
    x_others = torch.gather(image_features_b, dim=1, index=index_features)
    Key_others = torch.gather(desired_layer_k_b, dim=1, index=index_k)
    x_others_attn = torch.gather(cls_attn, dim=1, index=idx)
    compl = complement_idx_prumerge_plus(idx, N)
    non_topk = torch.gather(image_features_b, dim=1, index=compl.unsqueeze(-1).expand(-1, -1, C))
    non_topk_Key = torch.gather(desired_layer_k_b, dim=1, index=compl.unsqueeze(-1).expand(-1, -1, k_C))
    non_topk_attn = torch.gather(cls_attn, dim=1, index=compl)

    Key_others_norm = nn.functional.normalize(Key_others, p=2, dim=-1)
    non_topk_Key_norm = nn.functional.normalize(non_topk_Key, p=2, dim=-1)

    B_loc, left_tokens, C_loc = x_others.size()
    updated_x_others = torch.zeros_like(x_others)

    for b in range(B_loc):
        for i in range(left_tokens):
            key_n = Key_others_norm[b, i, :].unsqueeze(0).unsqueeze(0)
            before_k = Key_others_norm[b, :i, :].unsqueeze(0)
            after_k = Key_others_norm[b, i + 1:, :].unsqueeze(0)
            before_x = x_others[b, :i, :].unsqueeze(0)
            after_x = x_others[b, i + 1:, :].unsqueeze(0)
            rest_x = torch.cat([before_x, after_x, non_topk[b, :, :].unsqueeze(0)], dim=1)
            before_attn = x_others_attn[b, :i].unsqueeze(0)
            after_attn = x_others_attn[b, i + 1:].unsqueeze(0)
            rest_attn = torch.cat([before_attn, after_attn, non_topk_attn[b, :].unsqueeze(0)], dim=1)
            rest_keys = torch.cat([before_k, after_k, non_topk_Key_norm[b, :, :].unsqueeze(0)], dim=1)
            cos_sim = torch.bmm(key_n, rest_keys.transpose(1, 2))
            cos_sim_num = max(min(32, cos_sim.shape[2]), 1)
            _, cluster_idx = torch.topk(cos_sim, k=cos_sim_num, dim=2, largest=True)
            cluster_t = rest_x[:, cluster_idx.squeeze(), :]
            weights = rest_attn[:, cluster_idx.squeeze()].unsqueeze(-1)
            weighted_avg = torch.sum(cluster_t * weights, dim=1)
            updated_x_others[b, i, :] = x_others[b, i, :] + weighted_avg

    image_features = updated_x_others.squeeze(0)
    all_keep_indices = idx.squeeze(0).sort().values
    image_features = image_features.to(dtype=self.dtype)

    return image_features, all_keep_indices


# ---------------------------------------------------------------------------
# Prumerge+ — Generation forward (same structure as VisionZip)
# ---------------------------------------------------------------------------

def qwen2_5vl_generation_forward_prumerge_plus(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values=None,
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
    second_per_grid_ts: Optional[torch.Tensor] = None,
    logits_to_keep: int = 0,
    **kwargs,
):
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLCausalLMOutputWithPast
    from torch.nn import CrossEntropyLoss

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if inputs_embeds is None:
        inputs_embeds = self.model.get_input_embeddings()(input_ids)

        if pixel_values is not None:
            pixel_values = pixel_values.type(self.model.visual.dtype)
            image_embeds, all_indices = self.model.visual(pixel_values, grid_thw=image_grid_thw)
            n_image_tokens = image_embeds.shape[0]
            total_len = input_ids.shape[-1]
            assert input_ids.shape[0] == 1, "Prumerge+ only supports batch_size=1"

            vision_start_id = self.config.vision_start_token_id
            vision_end_id = self.config.vision_end_token_id

            position_image_begin = (input_ids[0] == vision_start_id).nonzero(as_tuple=True)[0]
            before_idx = position_image_begin[0].item() + 1
            before_img = input_ids[:, :before_idx]
            position_image_end = (input_ids[0] == vision_end_id).nonzero(as_tuple=True)[0]
            post_idx = position_image_end[-1].item()
            post_img = input_ids[:, post_idx:]

            img_tensor = torch.full(
                (input_ids.shape[0], n_image_tokens),
                self.config.image_token_id,
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            origin_input_ids = deepcopy(input_ids)
            input_ids = torch.cat((before_img, img_tensor, post_img), dim=1)
            all_indices = all_indices + before_idx
            all_indices = torch.cat((
                torch.arange(0, before_idx, device=all_indices.device),
                all_indices,
                torch.arange(post_idx, total_len, device=all_indices.device),
            ))
            inputs_embeds = inputs_embeds[:, all_indices, :]

            image_mask = (
                (input_ids == self.config.image_token_id)
                .unsqueeze(-1)
                .expand_as(inputs_embeds)
                .to(inputs_embeds.device)
            )
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            # Save all_indices for decode stage
            self.model.all_indices = all_indices

            # Save original attention_mask for get_rope_index (needed before slicing)
            _orig_attn_mask_for_rope = attention_mask
            _compressed_len = all_indices.shape[0]

            text_image_mask = (input_ids != self.config.image_token_id)
            self.model.language_model.text_image_mask = text_image_mask
            for layer in self.model.language_model.layers:
                layer.self_attn.text_image_mask = text_image_mask

        if pixel_values_videos is not None:
            pixel_values_videos = pixel_values_videos.type(self.model.visual.dtype)
            video_embeds = self.model.visual(pixel_values_videos, grid_thw=video_grid_thw)
            video_mask = (
                (input_ids == self.config.video_token_id)
                .unsqueeze(-1)
                .expand_as(inputs_embeds)
                .to(inputs_embeds.device)
            )
            video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        if attention_mask is not None:
            attention_mask = attention_mask.to(inputs_embeds.device)

        # Prumerge+ prefill: adjust positional information to match compressed sequence.
        if pixel_values is not None:
            if attention_mask is not None:
                attention_mask = attention_mask[:, all_indices]
            if position_ids is not None:
                position_ids = position_ids[:, :, all_indices]
            if self.model.rope_deltas is not None:
                self.model.rope_deltas = self.model.rope_deltas + (total_len - _compressed_len)

    if position_ids is None and input_ids is not None and (attention_mask is None or attention_mask.ndim == 2):
        if (cache_position is not None and cache_position[0] == 0) or self.model.rope_deltas is None:
            position_ids, rope_deltas = self.model.get_rope_index(
                origin_input_ids, image_grid_thw, video_grid_thw,
                second_per_grid_ts=second_per_grid_ts, attention_mask=_orig_attn_mask_for_rope,
            )
            self.model.rope_deltas = rope_deltas + (total_len - _compressed_len)
            position_ids = position_ids[:, :, all_indices]
            # Note: attention_mask was already sliced above
        else:
            # Decode stage: position_ids for the new token.
            batch_size_here, seq_length_here, _ = inputs_embeds.shape
            position_ids = torch.arange(seq_length_here, device=inputs_embeds.device)
            position_ids = position_ids.view(1, 1, -1).expand(3, batch_size_here, -1)
            if cache_position is not None and self.model.rope_deltas is not None:
                delta = (cache_position[0] + self.model.rope_deltas).to(inputs_embeds.device)
            else:
                delta = torch.zeros((batch_size_here, seq_length_here), device=inputs_embeds.device)
            delta = delta.repeat_interleave(batch_size_here // delta.shape[0], dim=0)
            position_ids = position_ids + delta.to(position_ids.device)

    outputs = self.model(
        input_ids=None,
        pixel_values=None,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=True,
        cache_position=cache_position,
    )

    hidden_states = outputs[0]
    logits = self.lm_head(hidden_states)

    loss = None
    if labels is not None:
        logits_f = logits.float()
        shift_logits = logits_f[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1).to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return Qwen2_5_VLCausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=self.model.rope_deltas,
    )


# ===========================================================================
# DART — Duplication-Aware Reduction of Tokens (Qwen2.5-VL)
# ===========================================================================

from .qwen2vl_model import get_retained_image_token_dart


# ---------------------------------------------------------------------------
# DART — Attention forward for Qwen2.5-VL (stores k_states at layer K-1)
# ---------------------------------------------------------------------------

def qwen2_5vl_flash_attention_forward_dart(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    **kwargs,
):
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
        apply_multimodal_rotary_pos_emb,
        ALL_ATTENTION_FUNCTIONS,
        eager_attention_forward,
    )
    from transformers.models.llama.modeling_llama import repeat_kv

    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_multimodal_rotary_pos_emb(
        query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
    )

    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

    # DART: store k_states (after RoPE + repeat_kv) at layer K-1
    if self.layer_idx == self.target_layer_idx - 1:
        self.last_k_states = repeat_kv(key_states, self.num_key_value_groups)

    attention_interface = eager_attention_forward
    if self.config._attn_implementation != "eager":
        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        sliding_window=getattr(self.config, "sliding_window", None),
        **kwargs,
    )

    attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights if output_attentions else None


# ---------------------------------------------------------------------------
# DART — Text Model forward for Qwen2.5-VL
# ---------------------------------------------------------------------------

def qwen2_5vl_text_model_forward_dart(
    self,
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
    **kwargs,
) -> Union[Tuple, BaseModelOutputWithPast]:
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
        create_causal_mask,
        create_sliding_window_causal_mask,
    )

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if self.gradient_checkpointing and self.training:
        use_cache = False

    if use_cache and past_key_values is None and not torch.jit.is_tracing():
        past_key_values = DynamicCache(config=self.config)

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    batch_size, seq_length = inputs_embeds.shape[:2]

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + seq_length, device=inputs_embeds.device
        )

    if position_ids is None:
        position_ids = cache_position.view(1, 1, -1).expand(3, batch_size, -1)
    elif position_ids.ndim == 2:
        position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

    if position_ids.ndim == 3 and position_ids.shape[0] == 4:
        text_position_ids = position_ids[0]
        position_ids = position_ids[1:]
    else:
        text_position_ids = None

    if not isinstance(causal_mask_mapping := attention_mask, dict):
        mask_kwargs = {
            "config": self.config,
            "input_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "cache_position": cache_position,
            "past_key_values": past_key_values,
            "position_ids": text_position_ids,
        }
        causal_mask_mapping = {"full_attention": create_causal_mask(**mask_kwargs)}
        if self.has_sliding_layers:
            causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

    hidden_states = inputs_embeds
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None

    for decoder_layer in self.layers:
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # DART pruning at target_layer_idx (prefill only)
        if decoder_layer.self_attn.layer_idx == self.target_layer_idx and seq_length > 1:
            assert batch_size == 1, "DART only supports batch_size=1"
            last_k_states = self.layers[self.target_layer_idx - 1].self_attn.last_k_states
            assert last_k_states is not None, "last_k_states is None at DART target layer"

            text_image_mask = self.text_image_mask[0]
            image_positions = (text_image_mask == False).nonzero(as_tuple=True)[0]
            assert len(image_positions) > 0, "No image tokens found; DART requires an image"
            image_start = int(image_positions[0])
            image_end = int(image_positions[-1])
            image_length = image_end - image_start + 1

            device = hidden_states.device
            last_layer_state = self.norm(hidden_states)

            retained = get_retained_image_token_dart(
                last_layer_state, last_k_states,
                image_start, image_length,
                self.budgets, self.pivot_image_token, self.pivot_text_token,
            )

            keep_indices = torch.cat((
                torch.arange(image_start, device=device),
                retained.to(device),
                torch.arange(image_start + image_length, seq_length, device=device),
            )).sort().values

            hidden_states = hidden_states[:, keep_indices, :]
            seq_length = hidden_states.shape[1]

            # Adjust causal masks
            new_mask_mapping = {}
            for k, v in causal_mask_mapping.items():
                if v is not None:
                    new_mask_mapping[k] = v[:, :, keep_indices, :][:, :, :, keep_indices]
                else:
                    new_mask_mapping[k] = None
            causal_mask_mapping = new_mask_mapping

            # Adjust position_ids and recompute rotary embeddings
            position_ids = position_ids[:, :, keep_indices]
            if text_position_ids is not None:
                text_position_ids = text_position_ids[:, keep_indices]
            cache_position = cache_position[keep_indices]
            position_embeddings = self.rotary_emb(hidden_states, position_ids)

        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=causal_mask_mapping[decoder_layer.attention_type],
            position_ids=text_position_ids,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )

        hidden_states = layer_outputs[0]
        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    if not return_dict:
        return tuple(v for v in [hidden_states, past_key_values, all_hidden_states, all_self_attns] if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )


# ---------------------------------------------------------------------------
# DART — Generation forward for Qwen2.5-VL
# Sets text_image_mask on the language model before calling self.model(...)
# ---------------------------------------------------------------------------

def qwen2_5vl_generation_forward_dart(
    self,
    input_ids=None,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    inputs_embeds=None,
    labels=None,
    use_cache=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict: Optional[bool] = None,
    pixel_values=None,
    pixel_values_videos=None,
    image_grid_thw=None,
    video_grid_thw=None,
    rope_deltas=None,
    cache_position=None,
    second_per_grid_ts=None,
    logits_to_keep=0,
    **kwargs,
):
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLCausalLMOutputWithPast
    from torch.nn import CrossEntropyLoss

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # Set text_image_mask for DART pruning (prefill stage only)
    if input_ids is not None and pixel_values is not None and inputs_embeds is None:
        text_image_mask = (input_ids != self.config.image_token_id)
        self.model.language_model.text_image_mask = text_image_mask
        for layer in self.model.language_model.layers:
            layer.self_attn.text_image_mask = text_image_mask

    outputs = self.model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        pixel_values_videos=pixel_values_videos,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        second_per_grid_ts=second_per_grid_ts,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=True,
        cache_position=cache_position,
        **kwargs,
    )

    hidden_states = outputs[0]
    logits = self.lm_head(hidden_states[:, -logits_to_keep:, :] if isinstance(logits_to_keep, int) and logits_to_keep > 0 else hidden_states)

    loss = None
    if labels is not None:
        logits_for_loss = logits.float()
        shift_logits = logits_for_loss[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1).to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return Qwen2_5_VLCausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=getattr(self.model, 'rope_deltas', None),
    )
