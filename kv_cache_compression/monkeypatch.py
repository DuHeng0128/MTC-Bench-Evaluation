      
import sys
import transformers
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention
from qwen2vl.modeling_qwen2_vl import Qwen2VLFlashAttention2, Qwen2VLModel
from transformers.models.qwen2.modeling_qwen2 import Qwen2Model
from .qwen_model import (
    qwen_attention_forward_streamingLLM,
    qwen_attention_forward_H2O,
    qwen_model_forward_vlcache,
    qwen_attention_forward_vlcache,
    qwen_attention_forward_LOOK_M,
    qwen_attention_forward_snapkv,
    qwen_model_forward_fastv,
    qwen_attention_forward_fastv,
    qwen_attn_forward_PyramidKV,
    qwen_attention_forward_CSP,
    qwen_attention_forward_random
)

from .qwen_model import (
    qwen_flash_attention_forward_streamingLLM,
    qwen_flash_attention_forward_H2O,
    qwen_flash_attention_forward_PyramidKV,
    qwen_flash_attention_forward_random,
    qwen_flash_attention_forward_snapkv,
    qwen2vl_model_forward_vlcache,
    qwen_flash_attention_forward_vlcache,
    qwen_flash_attention_forward_CSP
)

from .internlm2_model import (
    internlm2_flash_attention_forward_streamingLLM,

)
from .internvl2_5_model import (
    internvl_generate_4B_visionzip,
    internvl_extract_feature_4B_visionzip,
    internvl_attention_forward_4B_visionzip,
    internvl_naive_attn_4B_visionzip,
    internvl_generate_with_mask,
    internvl_generate_38B_visionzip,
    internvl_extract_feature_38B_visionzip,
    internvl_attention_forward_38B_visionzip,
    internvl_naive_attn_38B_visionzip,
    internvl_generate_4B_prumerge_plus,
    internvl_extract_feature_4B_prumerge_plus,
    internvl_attention_forward_4B_prumerge_plus,
    internvl_naive_attn_4B_prumerge_plus,
    internvl_generate_38B_prumerge_plus,
    internvl_extract_feature_38B_prumerge_plus,
    internvl_attention_forward_38B_prumerge_plus,
    internvl_naive_attn_38B_prumerge_plus,
)
import types

from .kv_cache_utils import VlCacheKVCluster
from .siglip_model import *
import qwen2vl
from .qwen2vl_model import (
    qwen_vl_model_forward_fastv,
    qwen_vl_flash_attention_forward_fastv,
    qwen2vl_vision_flash_attention2_forward_visionzip,
    qwen2vl_vision_tower_forward_visionzip,
    qwen2vl_vision_block_forward_visionzip,
    qwen2vl_generation_forward_visionzip,
    qwen2vl_vision_flash_attention2_forward_prumerge_plus,
    qwen2vl_vision_tower_forward_prumerge_plus,
    qwen2vl_vision_block_forward_prumerge_plus,
    qwen2vl_generation_forward_prumerge_plus,
    qwen_flash_attention_forward_look_m,
    qwen_vl_flash_attention_forward_dart,
    qwen_vl_model_forward_dart,
)
from .qwen2_5vl_model import (
    qwen2_5vl_flash_attention_forward_fastv,
    qwen2_5vl_text_model_forward_fastv,
    qwen2_5vl_generation_forward_fastv,
    qwen2_5vl_vision_attention_forward_visionzip,
    qwen2_5vl_vision_block_forward_visionzip,
    qwen2_5vl_vision_tower_forward_visionzip,
    qwen2_5vl_generation_forward_visionzip,
    qwen2_5vl_vision_attention_forward_prumerge_plus,
    qwen2_5vl_vision_block_forward_prumerge_plus,
    qwen2_5vl_vision_tower_forward_prumerge_plus,
    qwen2_5vl_generation_forward_prumerge_plus,
    qwen2_5vl_flash_attention_forward_dart,
    qwen2_5vl_text_model_forward_dart,
    qwen2_5vl_generation_forward_dart,
)


def replace_qwen(args, model, method):

    if method == "streamingllm":
        print('using streamingllm')
        for name, module in model.named_modules():
            if isinstance(module, Qwen2Attention):
                module.forward = types.MethodType(qwen_attention_forward_streamingLLM, module)
                module.budgets = args.budgets
    elif method == "h2o":
        print('using h2o')
        for name, module in model.named_modules():
            if isinstance(module, Qwen2Attention):
                module.forward = types.MethodType(qwen_attention_forward_H2O, module)
                module.budgets = args.budgets
                module.h2o_head_adaptive = args.h2o_head_adaptive

    elif method == "vl-cache":
        print('using vlcache')
        for name, module in model.named_modules():
            if isinstance(module, Qwen2Attention):
                module.forward = types.MethodType(qwen_attention_forward_vlcache, module)
                module.vlcache_alpha_sparsity = getattr(args,'budgets',1.0)
                module.vlcache_different_window_per_layer = getattr(args,'vlcache_different_window_per_layer',False)
                module.vlcache_head_adaptive = getattr(args,'head_adaptive',False)
                module.vlcache_budget_layer_adaptive = getattr(args, 'layer_adaptive', True)
            if isinstance(module, Qwen2Model):
                module.forward = types.MethodType(qwen_model_forward_vlcache, module)

    elif method == 'look-m':
        print('using look-m')
        for name, module in model.named_modules():
            if isinstance(module, Qwen2Attention):
                module.forward = types.MethodType(qwen_attention_forward_LOOK_M, module)
                module.budget = getattr(args, 'budgets', 1.0)
                module.merge = getattr(args, 'merge', False)
                module.hh_ratio = getattr(args, 'hh_ratio', None)
                module.recent_ratio = getattr(args, 'recent_ratio', None)

    elif method == 'snapkv':
        print('using snapkv')
        for name, module in model.named_modules():
            if isinstance(module, Qwen2Attention):
                module.forward = types.MethodType(qwen_attention_forward_snapkv, module)
                module.budgets = args.budgets
                module.snapkv_head_adaptive = args.snapkv_head_adaptive
                module.pooling = args.pooling

    elif method == 'fastv':
        print('using fastv')
        for name, module in model.named_modules():
            if isinstance(module, Qwen2Attention):
                module.forward = types.MethodType(qwen_attention_forward_fastv, module)
                module.target_layer_idx = getattr(args, 'target_layer_idx', 2)
            if isinstance(module, Qwen2Model):
                module.forward = types.MethodType(qwen_model_forward_fastv, module)
                module.target_layer_idx = getattr(args, 'target_layer_idx', 2)
                module.budgets = getattr(args, 'budgets', 1.0)
                module.origin = getattr(args, 'origin', False)

    # elif method == "csp":
    #     print('using csp')
    #     transformers.models.qwen2.modeling_qwen2.Qwen2Attention.forward = qwen_attention_forward_CSP
    #     transformers.models.qwen2.modeling_qwen2.Qwen2Attention.hh_ratio = getattr(
    #         args, 'hh_ratio', None)
    #     transformers.models.qwen2.modeling_qwen2.Qwen2Attention.recent_ratio = getattr(
    #         args, 'recent_ratio', None)
    #     # The budget ratio allocated to cross-attention. The default value is 0.1
    #     transformers.models.qwen2.modeling_qwen2.Qwen2Attention.cross_ratio = getattr(
    #         args, 'cross_ratio', 0.1)
    #     # The kv_recent_bias setting is how many times you want the recent window to be larger than the original window. The default value is 1, which means no additional increase in the window size.
    #     transformers.models.qwen2.modeling_qwen2.Qwen2Attention.kv_recent_bias = getattr(
    #         args, 'kv_recent_bias', 1)
    #     transformers.models.qwen2.modeling_qwen2.Qwen2Attention.budgets = getattr(
    #         args, 'budgets', None)
    #     transformers.models.qwen2.modeling_qwen2.Qwen2Attention.csp_head_adaptive = args.csp_head_adaptive

    elif method == 'pyramidkv':
        print('using pyramidkv')
        for name, module in model.named_modules():
            if isinstance(module, Qwen2Attention):
                module.forward = types.MethodType(qwen_attn_forward_PyramidKV, module)
                module.budgets = args.budgets
                module.pyramidkv_head_adaptive = args.pyramidkv_head_adaptive
                module.pooling = args.pooling
    
    elif method == 'random':
        print('using random')
        for name, module in model.named_modules():
            if isinstance(module, Qwen2Attention):
                module.forward = types.MethodType(qwen_attention_forward_random, module)
                module.budgets = getattr(args, 'budgets', 1.0)

    elif method == 'visionzip':
        print('using visionzip')
        from llava.model.multimodal_encoder.siglip_encoder import SigLipVisionTower, SigLipEncoderLayer, SigLipAttention
        from llava.model.llava_arch import LlavaMetaForCausalLM
        from llava.model.language_model.llava_qwen import LlavaQwenForCausalLM
        # SigLipVisionTower.dominant = getattr(args, 'dominant_num')
        # SigLipVisionTower.contextual = getattr(args, 'contextual_num')
        SigLipVisionTower.budgets = getattr(args, 'budgets', 1.0)
        SigLipVisionTower.forward = siglip_vision_tower_forward
        SigLipEncoderLayer.forward = siglip_EncoderLayer_forward
        SigLipAttention.forward = siglip_attention_forward
        LlavaMetaForCausalLM.encode_images_visionzip = encode_images_visionzip
        LlavaMetaForCausalLM.encode_images_visionzip_simple = encode_images_visionzip_simple
        LlavaQwenForCausalLM.prepare_inputs_labels_for_multimodal = prepare_inputs_labels_for_multimodal_visionzip


    elif method == 'prumerge+':
        print('using prumerge+')
        from llava.model.multimodal_encoder.siglip_encoder import SigLipVisionTower
        from llava.model.language_model.llava_qwen import LlavaQwenForCausalLM
        from llava.model.llava_arch import LlavaMetaForCausalLM
        SigLipVisionTower.budgets = getattr(args, 'budgets', 1.0)
        SigLipVisionTower.forward = siglip_vision_tower_forward_prumerge_plus
        LlavaMetaForCausalLM.encode_images_prumerge_plus = encode_images_prumerge_plus
        LlavaMetaForCausalLM.encode_images_prumerge_plus_simple = encode_images_prumerge_plus_simple
        LlavaQwenForCausalLM.prepare_inputs_labels_for_multimodal = prepare_inputs_labels_for_multimodal_prumerge_plus

    elif method == 'sparsevlm':
        print('using sparsevlm')
        from llava.model.language_model.llava_qwen import LlavaQwenForCausalLM
        from llava.model.language_model.sparse_llava_qwen import LlavaQwenSparseForCausalLM
        from llava.model.language_model.sparse_modeling_qwen import Qwen2SparseModel
        LlavaQwenSparseForCausalLM.bias = 0
        LlavaQwenSparseForCausalLM.scale = 13.5
        Qwen2SparseModel.ratio = getattr(args, 'r', 1.0)

    elif method == 'dart':
        print('using dart')
        from .qwen_model import qwen_attention_forward_dart, qwen_model_forward_dart
        target_layer_idx = int(getattr(args, 'target_layer_idx', 2))
        for name, module in model.named_modules():
            if isinstance(module, Qwen2Attention):
                module.forward = types.MethodType(qwen_attention_forward_dart, module)
                module.target_layer_idx = target_layer_idx
            if isinstance(module, Qwen2Model):
                module.forward = types.MethodType(qwen_model_forward_dart, module)
                module.target_layer_idx = target_layer_idx
                module.budgets = float(getattr(args, 'budgets', 1.0))
                module.pivot_image_token = int(getattr(args, 'pivot_image_token', 4))
                module.pivot_text_token = int(getattr(args, 'pivot_text_token', 4))


def replace_qwen2vl(args, model, method):

    if method == 'streamingllm':
        print('using streamingllm')
        for name, module in model.named_modules():
            if isinstance(module, Qwen2VLFlashAttention2):
                module.forward = types.MethodType(qwen_flash_attention_forward_streamingLLM, module)
                module.budgets = args.budgets

    elif method == "h2o":
        print('using h2o')
        for name, module in model.named_modules():
            if isinstance(module, Qwen2VLFlashAttention2):
                module.forward = types.MethodType(qwen_flash_attention_forward_H2O, module)
                module.budgets = args.budgets
                module.h2o_head_adaptive = args.h2o_head_adaptive

    elif method == 'snapkv':
        print('using snapkv')
        for name, module in model.named_modules():
            if isinstance(module, Qwen2VLFlashAttention2):
                module.forward = types.MethodType(qwen_flash_attention_forward_snapkv, module)
                module.budgets = args.budgets
                module.snapkv_head_adaptive = args.snapkv_head_adaptive
                module.pooling = args.pooling

    elif method == 'pyramidkv':
        print('using pyramidkv')
        for name, module in model.named_modules():
            if isinstance(module, Qwen2VLFlashAttention2):
                module.forward = types.MethodType(qwen_flash_attention_forward_PyramidKV, module)
                module.budgets = args.budgets
                module.pyramidkv_head_adaptive = args.pyramidkv_head_adaptive
                module.pooling = args.pooling

    elif method == 'random':
        print('using random')
        for name, module in model.named_modules():
            if isinstance(module, Qwen2VLFlashAttention2):
                module.forward = types.MethodType(qwen_flash_attention_forward_random, module)
                module.budgets = getattr(args, 'budgets', 1.0)

    elif method == 'look-m':
        print('using look-m')
        for name, module in model.named_modules():
            if isinstance(module, Qwen2VLFlashAttention2):
                module.forward = types.MethodType(qwen_flash_attention_forward_look_m, module)
                module.budget = getattr(args, 'budgets', 1.0)
                module.merge = getattr(args, 'merge', True)
                module.hh_ratio = getattr(args, 'hh_ratio', None)
                module.recent_ratio = getattr(args, 'recent_ratio', None)

    elif method == 'fastv':
        print('using fastv')
        for name, module in model.named_modules():
            if isinstance(module, Qwen2VLFlashAttention2):
                module.forward = types.MethodType(qwen_vl_flash_attention_forward_fastv, module)
                module.target_layer_idx = getattr(args, 'target_layer_idx', 2)
            if isinstance(module, Qwen2VLModel):
                module.forward = types.MethodType(qwen_vl_model_forward_fastv, module)
                module.target_layer_idx = getattr(args, 'target_layer_idx', 2)
                module.budgets = getattr(args, 'budgets', 1.0)
                module.origin = getattr(args, 'origin', False)

    elif method == 'visionzip':
        print('using visionzip')
        qwen2vl.modeling_qwen2_vl.VisionFlashAttention2.forward = qwen2vl_vision_flash_attention2_forward_visionzip
        qwen2vl.modeling_qwen2_vl.Qwen2VisionTransformerPretrainedModel.forward = qwen2vl_vision_tower_forward_visionzip
        qwen2vl.modeling_qwen2_vl.Qwen2VLVisionBlock.forward = qwen2vl_vision_block_forward_visionzip
        qwen2vl.modeling_qwen2_vl.Qwen2VisionTransformerPretrainedModel.budgets = getattr(
            args, 'budgets', 1.0)
        qwen2vl.modeling_qwen2_vl.Qwen2VLForConditionalGeneration.forward = qwen2vl_generation_forward_visionzip

    elif method == "vl-cache":
        print('using vlcache')
        for name, module in model.named_modules():
            if isinstance(module, Qwen2VLFlashAttention2):
                module.forward = types.MethodType(qwen_flash_attention_forward_vlcache, module)
                module.vlcache_alpha_sparsity = getattr(args,'budgets',1.0)
                module.vlcache_different_window_per_layer = getattr(args,'vlcache_different_window_per_layer',False)
                module.vlcache_head_adaptive = getattr(args,'head_adaptive',False)
                module.vlcache_budget_layer_adaptive = getattr(args, 'layer_adaptive', True)
            if isinstance(module, Qwen2VLModel):
                module.forward = types.MethodType(qwen2vl_model_forward_vlcache, module)

    # elif method == "csp":
    #     print('using csp')
    #     qwen2vl.modeling_qwen2_vl.Qwen2VLFlashAttention2.forward = qwen_flash_attention_forward_CSP
    #     qwen2vl.modeling_qwen2_vl.Qwen2VLFlashAttention2.hh_ratio = getattr(args, 'hh_ratio', None)
    #     qwen2vl.modeling_qwen2_vl.Qwen2VLFlashAttention2.recent_ratio = getattr(args, 'recent_ratio', None)
    #     qwen2vl.modeling_qwen2_vl.Qwen2VLFlashAttention2.cross_ratio = getattr(args, 'cross_ratio', 0.1)
    #     qwen2vl.modeling_qwen2_vl.Qwen2VLFlashAttention2.kv_recent_bias = getattr(args, 'kv_recent_bias', 1)
    #     qwen2vl.modeling_qwen2_vl.Qwen2VLFlashAttention2.budgets = getattr(args, 'budgets', None)
    #     qwen2vl.modeling_qwen2_vl.Qwen2VLFlashAttention2.csp_head_adaptive = args.csp_head_adaptive

    elif method == 'prumerge+':
        print('using prumerge+')
        qwen2vl.modeling_qwen2_vl.VisionFlashAttention2.forward = qwen2vl_vision_flash_attention2_forward_prumerge_plus
        qwen2vl.modeling_qwen2_vl.Qwen2VisionTransformerPretrainedModel.forward = qwen2vl_vision_tower_forward_prumerge_plus
        qwen2vl.modeling_qwen2_vl.Qwen2VLVisionBlock.forward = qwen2vl_vision_block_forward_prumerge_plus
        qwen2vl.modeling_qwen2_vl.Qwen2VisionTransformerPretrainedModel.budgets = getattr(args, 'budgets', 1.0)
        qwen2vl.modeling_qwen2_vl.Qwen2VLForConditionalGeneration.forward = qwen2vl_generation_forward_prumerge_plus

    elif method == 'dart':
        print('using dart')
        from qwen2vl.modeling_qwen2_vl import Qwen2VLFlashAttention2 as Qwen2VLFlashAttention2_cls
        from qwen2vl.modeling_qwen2_vl import Qwen2VLModel as Qwen2VLModel_cls
        target_layer_idx = int(getattr(args, 'target_layer_idx', 2))
        for name, module in model.named_modules():
            if isinstance(module, Qwen2VLFlashAttention2_cls):
                module.forward = types.MethodType(qwen_vl_flash_attention_forward_dart, module)
                module.target_layer_idx = target_layer_idx
            if isinstance(module, Qwen2VLModel_cls):
                module.forward = types.MethodType(qwen_vl_model_forward_dart, module)
                module.target_layer_idx = target_layer_idx
                module.budgets = float(getattr(args, 'budgets', 1.0))
                module.pivot_image_token = int(getattr(args, 'pivot_image_token', 4))
                module.pivot_text_token = int(getattr(args, 'pivot_text_token', 4))

def replace_internvl2_5(args, model, method):

    module_name = model.__class__.__module__
    if '8B' in module_name and "38B" not in module_name:
        mod = sys.modules.get(
            'transformers_modules.InternVL2_5-8B.modeling_internlm2', None)
    elif '26B' in module_name:
        mod = sys.modules.get(
            'transformers_modules.InternVL2_5-26B.modeling_internlm2', None)
    elif '38B' in module_name:
        return replace_qwen_for_internvl_38B(args, model, method)
    else:
        return replace_qwen_for_internvl(args, model, method)

    raise NotImplementedError('Not implemented yet')
    if method == 'streamingllm':
        print('using streamingllm')

        InternLM2AttnClass = getattr(mod, "InternLM2FlashAttention2", None)
        InternLM2AttnClass.forward = internlm2_flash_attention_forward_streamingLLM
        InternLM2AttnClass.budgets = args.budgets



def replace_qwen_for_internvl(args, model, method):

    module_name = model.__class__.__module__
    if '4B' in module_name:
        mod = sys.modules.get(
            'transformers_modules.InternVL2_5-4B.modeling_internvl_chat', None)
        InternVLChatModel = mod.InternVLChatModel
        model.generate = types.MethodType(internvl_generate_with_mask, model)

    if method == "streamingllm":
        print('using streamingllm')
        transformers.models.qwen2.modeling_qwen2.Qwen2Attention.forward = qwen_attention_forward_streamingLLM
        transformers.models.qwen2.modeling_qwen2.Qwen2Attention.budgets = args.budgets
    elif method == "h2o":
        print('using h2o')
        transformers.models.qwen2.modeling_qwen2.Qwen2Attention.forward = qwen_attention_forward_H2O
        transformers.models.qwen2.modeling_qwen2.Qwen2Attention.budgets = args.budgets
        transformers.models.qwen2.modeling_qwen2.Qwen2Attention.h2o_head_adaptive = args.h2o_head_adaptive

    elif method == "vl-cache":
        print('using vlcache')
        transformers.models.qwen2.modeling_qwen2.Qwen2Attention.forward = qwen_attention_forward_vlcache
        transformers.models.qwen2.modeling_qwen2.Qwen2Model.forward = qwen_model_forward_vlcache
        transformers.models.qwen2.modeling_qwen2.Qwen2Attention.vlcache_alpha_sparsity = getattr(args,'budgets',1.0)
        transformers.models.qwen2.modeling_qwen2.Qwen2Attention.vlcache_different_window_per_layer = getattr(args,'vlcache_different_window_per_layer',False)
        transformers.models.qwen2.modeling_qwen2.Qwen2Attention.vlcache_head_adaptive = getattr(args,'head_adaptive',False)
        transformers.models.qwen2.modeling_qwen2.Qwen2Attention.vlcache_budget_layer_adaptive = getattr(args, 'layer_adaptive', True)

    elif method == 'look-m':
        print('using look-m')
        transformers.models.qwen2.modeling_qwen2.Qwen2Attention.hh_ratio = getattr(
            args, 'hh_ratio', None)
        transformers.models.qwen2.modeling_qwen2.Qwen2Attention.recent_ratio = getattr(
            args, 'recent_ratio', None)
        transformers.models.qwen2.modeling_qwen2.Qwen2Attention.budget = getattr(
            args, 'budgets', 1.0)
        transformers.models.qwen2.modeling_qwen2.Qwen2Attention.merge = getattr(
            args, 'merge', True)
        transformers.models.qwen2.modeling_qwen2.Qwen2Attention.forward = qwen_attention_forward_LOOK_M

    elif method == 'snapkv':
        print('using snapkv')
        transformers.models.qwen2.modeling_qwen2.Qwen2Attention.forward = qwen_attention_forward_snapkv
        transformers.models.qwen2.modeling_qwen2.Qwen2Attention.budgets = args.budgets
        transformers.models.qwen2.modeling_qwen2.Qwen2Attention.snapkv_head_adaptive = args.snapkv_head_adaptive
        transformers.models.qwen2.modeling_qwen2.Qwen2Attention.pooling = args.pooling

    elif method == 'fastv':
        print('using fastv')
        transformers.models.qwen2.modeling_qwen2.Qwen2Model.forward = qwen_model_forward_fastv
        transformers.models.qwen2.modeling_qwen2.Qwen2Attention.forward = qwen_attention_forward_fastv
        transformers.models.qwen2.modeling_qwen2.Qwen2Attention.target_layer_idx = getattr(
            args, 'target_layer_idx', 2)
        transformers.models.qwen2.modeling_qwen2.Qwen2Model.target_layer_idx = getattr(
            args, 'target_layer_idx', 2)
        transformers.models.qwen2.modeling_qwen2.Qwen2Model.budgets = getattr(
            args, 'budgets', 1.0)   # visual part
        transformers.models.qwen2.modeling_qwen2.Qwen2Model.origin = getattr(
            args, 'origin', False)

    elif method == 'pyramidkv':
        print('using pyramidkv')
        transformers.models.qwen2.modeling_qwen2.Qwen2Attention.forward = qwen_attn_forward_PyramidKV
        transformers.models.qwen2.modeling_qwen2.Qwen2Attention.budgets = args.budgets
        transformers.models.qwen2.modeling_qwen2.Qwen2Attention.pyramidkv_head_adaptive = args.pyramidkv_head_adaptive
        transformers.models.qwen2.modeling_qwen2.Qwen2Attention.pooling = args.pooling

    elif method == 'visionzip':
        print('using visionzip')
        for idx, layer in enumerate(model.vision_model.encoder.layers):  
            layer.attn.layer_idx = idx
            layer.attn.forward = types.MethodType(internvl_attention_forward_4B_visionzip, layer.attn)
            layer.attn._naive_attn = types.MethodType(internvl_naive_attn_4B_visionzip, layer.attn)
        model.generate = types.MethodType(internvl_generate_4B_visionzip, model)
        model.extract_feature = types.MethodType(internvl_extract_feature_4B_visionzip, model)
        model.budgets = getattr(args, 'budgets', 1.0)

    elif method == 'random':
        print('using random')
        transformers.models.qwen2.modeling_qwen2.Qwen2Attention.budgets = getattr(
            args, 'budgets', 1.0)
        transformers.models.qwen2.modeling_qwen2.Qwen2Attention.forward = qwen_attention_forward_random

    elif method == 'prumerge+':
        print('using prumerge+')
        for idx, layer in enumerate(model.vision_model.encoder.layers): 
            layer.attn.layer_idx = idx
            layer.attn.forward = types.MethodType(internvl_attention_forward_4B_prumerge_plus, layer.attn)
            layer.attn._naive_attn = types.MethodType(internvl_naive_attn_4B_prumerge_plus, layer.attn)
        model.generate = types.MethodType(internvl_generate_4B_prumerge_plus, model)
        model.extract_feature = types.MethodType(internvl_extract_feature_4B_prumerge_plus, model)
        model.budgets = getattr(args, 'budgets', None)



def replace_qwen_for_internvl_38B(args, model, method):

    module_name = model.__class__.__module__
    if '38B' in module_name:
        mod = sys.modules.get(
            'transformers_modules.InternVL2_5-38B.modeling_internvl_chat', None)
        InternVLChatModel = mod.InternVLChatModel
        model.generate = types.MethodType(internvl_generate_with_mask, model)


    if method == "streamingllm":
        print('using streamingllm')
        for name, module in model.named_modules():
            if isinstance(module, Qwen2Attention):
                module.forward = types.MethodType(qwen_attention_forward_streamingLLM, module)
                module.budgets = args.budgets
    
    elif method == "h2o":
        print('using h2o')
        for name, module in model.named_modules():
            if isinstance(module, Qwen2Attention):
                module.forward = types.MethodType(qwen_attention_forward_H2O, module)
                module.budgets = args.budgets
                module.h2o_head_adaptive = args.h2o_head_adaptive

    elif method == "vl-cache":
        print('using vlcache')
        for name, module in model.named_modules():
            if isinstance(module, Qwen2Attention):
                module.forward = types.MethodType(qwen_attention_forward_vlcache, module)
                module.vlcache_alpha_sparsity = getattr(args,'budgets',1.0)
                module.vlcache_different_window_per_layer = getattr(args,'vlcache_different_window_per_layer',False)
                module.vlcache_head_adaptive = getattr(args,'head_adaptive',False)
                module.vlcache_budget_layer_adaptive = getattr(args, 'layer_adaptive', True)
            if isinstance(module, Qwen2Model):
                module.forward = types.MethodType(qwen_model_forward_vlcache, module)

    elif method == 'look-m':
        print('using look-m')
        for name, module in model.named_modules():
            if isinstance(module, Qwen2Attention):
                module.forward = types.MethodType(qwen_attention_forward_LOOK_M, module)
                module.hh_ratio = getattr(args, 'hh_ratio', None)
                module.recent_ratio = getattr(args, 'recent_ratio', None)
                module.budget = getattr(args, 'budgets', 1.0)
                module.merge = getattr(args, 'merge', True)

    elif method == 'snapkv':
        print('using snapkv')
        for name, module in model.named_modules():
            if isinstance(module, Qwen2Attention):
                module.forward = types.MethodType(qwen_attention_forward_snapkv, module)
                module.snapkv_head_adaptive = args.snapkv_head_adaptive
                module.pooling = args.pooling
                module.budgets = args.budgets


    elif method == 'fastv':
        print('using fastv')
        for name, module in model.named_modules():
            if isinstance(module, Qwen2Attention):
                module.forward = types.MethodType(qwen_attention_forward_fastv, module)
                module.target_layer_idx = getattr(args, 'target_layer_idx', 2)
            if isinstance(module, Qwen2Model):
                module.forward = types.MethodType(qwen_model_forward_fastv, module)
                module.target_layer_idx = getattr(args, 'target_layer_idx', 2)
                module.budgets = getattr(args, 'budgets', 1.0)
                module.origin = getattr(args, 'origin', False)


    elif method == 'pyramidkv':
        print('using pyramidkv')
        for name, module in model.named_modules():
            if isinstance(module, Qwen2Attention):
                module.forward = types.MethodType(qwen_attn_forward_PyramidKV, module)
                module.budgets = args.budgets
                module.pyramidkv_head_adaptive = args.pyramidkv_head_adaptive
                module.pooling = args.pooling

    
    elif method == 'random':
        print('using random')
        for name, module in model.named_modules():
            if isinstance(module, Qwen2Attention):
                module.budgets = getattr(args, 'budgets', 1.0)
                module.forward = types.MethodType(qwen_attention_forward_random, module)
    
    elif method == 'visionzip':
        print('using visionzip')
        for idx, layer in enumerate(model.vision_model.encoder.layers): 
            layer.attn.layer_idx = idx
            layer.attn.forward = types.MethodType(internvl_attention_forward_38B_visionzip, layer.attn)
            layer.attn._naive_attn = types.MethodType(internvl_naive_attn_38B_visionzip, layer.attn)
        model.generate = types.MethodType(internvl_generate_38B_visionzip, model)
        model.extract_feature = types.MethodType(internvl_extract_feature_38B_visionzip, model)
        model.budgets = getattr(args, 'budgets', 1.0)


    elif method == 'prumerge+':
        print('using prumerge+')
        for idx, layer in enumerate(model.vision_model.encoder.layers):
            layer.attn.layer_idx = idx
            layer.attn.forward = types.MethodType(internvl_attention_forward_38B_prumerge_plus, layer.attn)
            layer.attn._naive_attn = types.MethodType(internvl_naive_attn_38B_prumerge_plus, layer.attn)
        model.generate = types.MethodType(internvl_generate_38B_prumerge_plus, model)
        model.extract_feature = types.MethodType(internvl_extract_feature_38B_prumerge_plus, model)
        model.budgets = getattr(args, 'budgets', 1.0)

def replace_qwen2_5vl(args, model, method):
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLAttention

    if method == 'streamingllm':
        print('using streamingllm')
        for name, module in model.named_modules():
            if isinstance(module, Qwen2_5_VLAttention):
                module.forward = types.MethodType(qwen_flash_attention_forward_streamingLLM, module)
                module.budgets = args.budgets

    elif method == 'h2o':
        print('using h2o')
        for name, module in model.named_modules():
            if isinstance(module, Qwen2_5_VLAttention):
                module.forward = types.MethodType(qwen_flash_attention_forward_H2O, module)
                module.budgets = args.budgets
                module.h2o_head_adaptive = args.h2o_head_adaptive

    elif method == 'snapkv':
        print('using snapkv')
        for name, module in model.named_modules():
            if isinstance(module, Qwen2_5_VLAttention):
                module.forward = types.MethodType(qwen_flash_attention_forward_snapkv, module)
                module.budgets = args.budgets
                module.snapkv_head_adaptive = args.snapkv_head_adaptive
                module.pooling = args.pooling

    elif method == 'pyramidkv':
        print('using pyramidkv')
        for name, module in model.named_modules():
            if isinstance(module, Qwen2_5_VLAttention):
                module.forward = types.MethodType(qwen_flash_attention_forward_PyramidKV, module)
                module.budgets = args.budgets
                module.pyramidkv_head_adaptive = args.pyramidkv_head_adaptive
                module.pooling = args.pooling

    elif method == 'random':
        print('using random')
        for name, module in model.named_modules():
            if isinstance(module, Qwen2_5_VLAttention):
                module.forward = types.MethodType(qwen_flash_attention_forward_random, module)
                module.budgets = getattr(args, 'budgets', 1.0)

    elif method == 'look-m':
        print('using look-m')
        for name, module in model.named_modules():
            if isinstance(module, Qwen2_5_VLAttention):
                module.forward = types.MethodType(qwen_flash_attention_forward_look_m, module)
                module.budget = getattr(args, 'budgets', 1.0)
                module.merge = getattr(args, 'merge', True)
                module.hh_ratio = getattr(args, 'hh_ratio', None)
                module.recent_ratio = getattr(args, 'recent_ratio', None)

    elif method == 'vl-cache':
        print('using vlcache')
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLModel
        for name, module in model.named_modules():
            if isinstance(module, Qwen2_5_VLAttention):
                module.forward = types.MethodType(qwen_flash_attention_forward_vlcache, module)
                module.vlcache_alpha_sparsity = getattr(args, 'budgets', 1.0)
                module.vlcache_different_window_per_layer = getattr(args, 'vlcache_different_window_per_layer', False)
                module.vlcache_head_adaptive = getattr(args, 'head_adaptive', False)
                module.vlcache_budget_layer_adaptive = getattr(args, 'layer_adaptive', True)
            if isinstance(module, Qwen2_5_VLModel):
                module.forward = types.MethodType(qwen2vl_model_forward_vlcache, module)

    elif method == 'fastv':
        print('using fastv')
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
            Qwen2_5_VLTextModel, Qwen2_5_VLForConditionalGeneration,
        )
        target_layer_idx = getattr(args, 'target_layer_idx', 2)
        for name, module in model.named_modules():
            if isinstance(module, Qwen2_5_VLAttention):
                module.forward = types.MethodType(qwen2_5vl_flash_attention_forward_fastv, module)
                module.target_layer_idx = target_layer_idx
            if isinstance(module, Qwen2_5_VLTextModel):
                module.forward = types.MethodType(qwen2_5vl_text_model_forward_fastv, module)
                module.target_layer_idx = target_layer_idx
                module.budgets = getattr(args, 'budgets', 1.0)
                module.origin = getattr(args, 'origin', False)
        import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl as q25vl
        q25vl.Qwen2_5_VLForConditionalGeneration.forward = qwen2_5vl_generation_forward_fastv

    elif method == 'visionzip':
        print('using visionzip')
        import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl as q25vl
        n_blocks = len(model.visual.blocks)
        # Second-to-last fullatt block: VisionZip paper uses second-to-last layer for scoring.
        # block[-1] (last fullatt, e.g. 31) is specialised for merger input; block[-2] (e.g. 23)
        # gives more discriminative global attention for text / OCR content.
        fullatt_indexes = getattr(model.visual, 'fullatt_block_indexes', None)
        if fullatt_indexes and len(fullatt_indexes) >= 2:
            target_block_idx = fullatt_indexes[-2]
        elif fullatt_indexes:
            target_block_idx = fullatt_indexes[-1]
        else:
            target_block_idx = n_blocks - 2
        for idx, blk in enumerate(model.visual.blocks):
            blk.attn.layer_idx = idx
            blk.attn.target_layer_idx = target_block_idx
            blk.attn.forward = types.MethodType(qwen2_5vl_vision_attention_forward_visionzip, blk.attn)
            blk.forward = types.MethodType(qwen2_5vl_vision_block_forward_visionzip, blk)
        q25vl.Qwen2_5_VisionTransformerPretrainedModel.forward = qwen2_5vl_vision_tower_forward_visionzip
        q25vl.Qwen2_5_VisionTransformerPretrainedModel.budgets = getattr(args, 'budgets', 1.0)
        q25vl.Qwen2_5_VLForConditionalGeneration.forward = qwen2_5vl_generation_forward_visionzip

    elif method == 'prumerge+':
        print('using prumerge+')
        import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl as q25vl
        n_blocks = len(model.visual.blocks)
        fullatt_indexes = getattr(model.visual, 'fullatt_block_indexes', None)
        if fullatt_indexes and len(fullatt_indexes) >= 2:
            target_block_idx = fullatt_indexes[-2]
        elif fullatt_indexes:
            target_block_idx = fullatt_indexes[-1]
        else:
            target_block_idx = n_blocks - 2
        for idx, blk in enumerate(model.visual.blocks):
            blk.attn.layer_idx = idx
            blk.attn.target_layer_idx = target_block_idx
            blk.attn.forward = types.MethodType(qwen2_5vl_vision_attention_forward_prumerge_plus, blk.attn)
            blk.forward = types.MethodType(qwen2_5vl_vision_block_forward_prumerge_plus, blk)
        q25vl.Qwen2_5_VisionTransformerPretrainedModel.forward = qwen2_5vl_vision_tower_forward_prumerge_plus
        q25vl.Qwen2_5_VisionTransformerPretrainedModel.budgets = getattr(args, 'budgets', 1.0)
        q25vl.Qwen2_5_VLForConditionalGeneration.forward = qwen2_5vl_generation_forward_prumerge_plus

    elif method == 'dart':
        print('using dart')
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
            Qwen2_5_VLTextModel, Qwen2_5_VLForConditionalGeneration,
        )
        import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl as q25vl
        target_layer_idx = int(getattr(args, 'target_layer_idx', 2))
        for name, module in model.named_modules():
            if isinstance(module, Qwen2_5_VLAttention):
                module.forward = types.MethodType(qwen2_5vl_flash_attention_forward_dart, module)
                module.target_layer_idx = target_layer_idx
            if isinstance(module, Qwen2_5_VLTextModel):
                module.forward = types.MethodType(qwen2_5vl_text_model_forward_dart, module)
                module.target_layer_idx = target_layer_idx
                module.budgets = float(getattr(args, 'budgets', 1.0))
                module.pivot_image_token = int(getattr(args, 'pivot_image_token', 4))
                module.pivot_text_token = int(getattr(args, 'pivot_text_token', 4))
        q25vl.Qwen2_5_VLForConditionalGeneration.forward = qwen2_5vl_generation_forward_dart

    else:
        raise ValueError(f"Unknown method '{method}' for replace_qwen2_5vl")


def replace_qwen3vl(args, model, method):
    try:
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextAttention
    except ImportError:
        raise ImportError(
            "Qwen3VL is not available in the current transformers version. "
            "Please upgrade transformers to a version that includes Qwen3VL support."
        )

    if method == 'streamingllm':
        print('using streamingllm')
        for name, module in model.named_modules():
            if isinstance(module, Qwen3VLTextAttention):
                module.forward = types.MethodType(qwen_flash_attention_forward_streamingLLM, module)
                module.budgets = args.budgets

    elif method == 'h2o':
        print('using h2o')
        for name, module in model.named_modules():
            if isinstance(module, Qwen3VLTextAttention):
                module.forward = types.MethodType(qwen_flash_attention_forward_H2O, module)
                module.budgets = args.budgets
                module.h2o_head_adaptive = args.h2o_head_adaptive

    elif method == 'snapkv':
        print('using snapkv')
        for name, module in model.named_modules():
            if isinstance(module, Qwen3VLTextAttention):
                module.forward = types.MethodType(qwen_flash_attention_forward_snapkv, module)
                module.budgets = args.budgets
                module.snapkv_head_adaptive = args.snapkv_head_adaptive
                module.pooling = args.pooling

    elif method == 'pyramidkv':
        print('using pyramidkv')
        for name, module in model.named_modules():
            if isinstance(module, Qwen3VLTextAttention):
                module.forward = types.MethodType(qwen_flash_attention_forward_PyramidKV, module)
                module.budgets = args.budgets
                module.pyramidkv_head_adaptive = args.pyramidkv_head_adaptive
                module.pooling = args.pooling

    elif method == 'random':
        print('using random')
        for name, module in model.named_modules():
            if isinstance(module, Qwen3VLTextAttention):
                module.forward = types.MethodType(qwen_flash_attention_forward_random, module)
                module.budgets = getattr(args, 'budgets', 1.0)

    elif method == 'look-m':
        print('using look-m')
        for name, module in model.named_modules():
            if isinstance(module, Qwen3VLTextAttention):
                module.forward = types.MethodType(qwen_flash_attention_forward_look_m, module)
                module.budget = getattr(args, 'budgets', 1.0)
                module.merge = getattr(args, 'merge', True)
                module.hh_ratio = getattr(args, 'hh_ratio', None)
                module.recent_ratio = getattr(args, 'recent_ratio', None)

    elif method == 'vl-cache':
        print('using vlcache')
        try:
            from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel
        except ImportError:
            raise ImportError("Qwen3VLModel not found in current transformers version.")
        for name, module in model.named_modules():
            if isinstance(module, Qwen3VLTextAttention):
                module.forward = types.MethodType(qwen_flash_attention_forward_vlcache, module)
                module.vlcache_alpha_sparsity = getattr(args, 'budgets', 1.0)
                module.vlcache_different_window_per_layer = getattr(args, 'vlcache_different_window_per_layer', False)
                module.vlcache_head_adaptive = getattr(args, 'head_adaptive', False)
                module.vlcache_budget_layer_adaptive = getattr(args, 'layer_adaptive', True)
            if isinstance(module, Qwen3VLModel):
                module.forward = types.MethodType(qwen2vl_model_forward_vlcache, module)

    elif method == 'fastv':
        print('using fastv')
        try:
            from transformers.models.qwen3_vl.modeling_qwen3_vl import (
                Qwen3VLTextModel, Qwen3VLForConditionalGeneration,
            )
        except ImportError:
            raise ImportError("Qwen3VL text model classes not found; upgrade transformers.")
        target_layer_idx = getattr(args, 'target_layer_idx', 2)
        for name, module in model.named_modules():
            if isinstance(module, Qwen3VLTextAttention):
                module.forward = types.MethodType(qwen2_5vl_flash_attention_forward_fastv, module)
                module.target_layer_idx = target_layer_idx
            if isinstance(module, Qwen3VLTextModel):
                module.forward = types.MethodType(qwen2_5vl_text_model_forward_fastv, module)
                module.target_layer_idx = target_layer_idx
                module.budgets = getattr(args, 'budgets', 1.0)
                module.origin = getattr(args, 'origin', False)
        try:
            import transformers.models.qwen3_vl.modeling_qwen3_vl as q3vl
            q3vl.Qwen3VLForConditionalGeneration.forward = qwen2_5vl_generation_forward_fastv
        except ImportError:
            raise ImportError("Qwen3VL ForConditionalGeneration not found; upgrade transformers.")

    elif method == 'visionzip':
        print('using visionzip')
        try:
            import transformers.models.qwen3_vl.modeling_qwen3_vl as q3vl
        except ImportError:
            raise ImportError("Qwen3VL module not found; upgrade transformers.")
        n_blocks = len(model.visual.blocks)
        fullatt_indexes = getattr(model.visual, 'fullatt_block_indexes', None)
        if fullatt_indexes and len(fullatt_indexes) >= 2:
            target_block_idx = fullatt_indexes[-2]
        elif fullatt_indexes:
            target_block_idx = fullatt_indexes[-1]
        else:
            target_block_idx = n_blocks - 2
        for idx, blk in enumerate(model.visual.blocks):
            blk.attn.layer_idx = idx
            blk.attn.target_layer_idx = target_block_idx
            blk.attn.forward = types.MethodType(qwen2_5vl_vision_attention_forward_visionzip, blk.attn)
            blk.forward = types.MethodType(qwen2_5vl_vision_block_forward_visionzip, blk)
        q3vl.Qwen3VLVisionModel.forward = qwen2_5vl_vision_tower_forward_visionzip
        q3vl.Qwen3VLVisionModel.budgets = getattr(args, 'budgets', 1.0)
        q3vl.Qwen3VLForConditionalGeneration.forward = qwen2_5vl_generation_forward_visionzip

    elif method == 'prumerge+':
        print('using prumerge+')
        try:
            import transformers.models.qwen3_vl.modeling_qwen3_vl as q3vl
        except ImportError:
            raise ImportError("Qwen3VL module not found; upgrade transformers.")
        n_blocks = len(model.visual.blocks)
        fullatt_indexes = getattr(model.visual, 'fullatt_block_indexes', None)
        if fullatt_indexes and len(fullatt_indexes) >= 2:
            target_block_idx = fullatt_indexes[-2]
        elif fullatt_indexes:
            target_block_idx = fullatt_indexes[-1]
        else:
            target_block_idx = n_blocks - 2
        for idx, blk in enumerate(model.visual.blocks):
            blk.attn.layer_idx = idx
            blk.attn.target_layer_idx = target_block_idx
            blk.attn.forward = types.MethodType(qwen2_5vl_vision_attention_forward_prumerge_plus, blk.attn)
            blk.forward = types.MethodType(qwen2_5vl_vision_block_forward_prumerge_plus, blk)
        q3vl.Qwen3VLVisionModel.forward = qwen2_5vl_vision_tower_forward_prumerge_plus
        q3vl.Qwen3VLVisionModel.budgets = getattr(args, 'budgets', 1.0)
        q3vl.Qwen3VLForConditionalGeneration.forward = qwen2_5vl_generation_forward_prumerge_plus

    else:
        raise ValueError(f"Unknown method '{method}' for replace_qwen3vl")


def replace_mistral(method):
    pass


def replace_llama(method):
    pass


def replace_internvl3(args, model, method):
    """
    InternVL3 uses InternViT vision encoder + Qwen2.5 LLM backbone.
    Vision patches: reuse InternVL2.5-4B visionzip/prumerge+ functions (same InternViT).
    Text patches: use Qwen2.5-VL text attention functions.
    FastV patches: use Qwen2.5-VL fastv functions on the Qwen2 text attention.
    """
    # Find the trust_remote_code module for this model
    model_key = None
    for key in sys.modules:
        if 'InternVL3' in key and 'modeling_internvl_chat' in key:
            model_key = key
            break

    if model_key:
        model.generate = types.MethodType(internvl_generate_with_mask, model)

    if method in ('streamingllm', 'h2o', 'snapkv', 'pyramidkv', 'random', 'look-m', 'vl-cache', 'fastv'):
        # InternVL3 uses Qwen2 or Qwen2.5 as LLM; try Qwen2Attention first, then Qwen2_5_VLAttention
        # Determine which attention class is present
        from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention as _Qwen2Attn
        try:
            from transformers.models.qwen2_5.modeling_qwen2_5 import Qwen2_5Attention as _Qwen25Attn
            _has_qwen25 = True
        except ImportError:
            _has_qwen25 = False

        # Check if model uses Qwen2 or Qwen2.5 attention by inspecting modules
        _attn_class = None
        for _name, _mod in model.named_modules():
            if isinstance(_mod, _Qwen2Attn):
                _attn_class = _Qwen2Attn
                break
            if _has_qwen25 and isinstance(_mod, _Qwen25Attn):
                _attn_class = _Qwen25Attn
                break

        if method == 'streamingllm':
            print('using streamingllm')
            for name, module in model.named_modules():
                if _attn_class and isinstance(module, _attn_class):
                    module.forward = types.MethodType(qwen_attention_forward_streamingLLM, module)
                    module.budgets = args.budgets
        elif method == 'h2o':
            print('using h2o')
            for name, module in model.named_modules():
                if _attn_class and isinstance(module, _attn_class):
                    module.forward = types.MethodType(qwen_attention_forward_H2O, module)
                    module.budgets = args.budgets
                    module.h2o_head_adaptive = args.h2o_head_adaptive
        elif method == 'snapkv':
            print('using snapkv')
            for name, module in model.named_modules():
                if _attn_class and isinstance(module, _attn_class):
                    module.forward = types.MethodType(qwen_attention_forward_snapkv, module)
                    module.budgets = args.budgets
                    module.snapkv_head_adaptive = args.snapkv_head_adaptive
                    module.pooling = args.pooling
        elif method == 'pyramidkv':
            print('using pyramidkv')
            for name, module in model.named_modules():
                if _attn_class and isinstance(module, _attn_class):
                    module.forward = types.MethodType(qwen_attn_forward_PyramidKV, module)
                    module.budgets = args.budgets
                    module.pyramidkv_head_adaptive = args.pyramidkv_head_adaptive
                    module.pooling = args.pooling
        elif method == 'random':
            print('using random')
            for name, module in model.named_modules():
                if _attn_class and isinstance(module, _attn_class):
                    module.forward = types.MethodType(qwen_attention_forward_random, module)
                    module.budgets = getattr(args, 'budgets', 1.0)
        elif method == 'look-m':
            print('using look-m')
            for name, module in model.named_modules():
                if _attn_class and isinstance(module, _attn_class):
                    module.forward = types.MethodType(qwen_attention_forward_LOOK_M, module)
                    module.hh_ratio = getattr(args, 'hh_ratio', None)
                    module.recent_ratio = getattr(args, 'recent_ratio', None)
                    module.budget = getattr(args, 'budgets', 1.0)
                    module.merge = getattr(args, 'merge', True)
        elif method == 'vl-cache':
            print('using vlcache')
            from transformers.models.qwen2.modeling_qwen2 import Qwen2Model as _Q2Model
            for name, module in model.named_modules():
                if _attn_class and isinstance(module, _attn_class):
                    module.forward = types.MethodType(qwen_attention_forward_vlcache, module)
                    module.vlcache_alpha_sparsity = getattr(args, 'budgets', 1.0)
                    module.vlcache_different_window_per_layer = getattr(args, 'vlcache_different_window_per_layer', False)
                    module.vlcache_head_adaptive = getattr(args, 'head_adaptive', False)
                    module.vlcache_budget_layer_adaptive = getattr(args, 'layer_adaptive', True)
                if isinstance(module, _Q2Model):
                    module.forward = types.MethodType(qwen_model_forward_vlcache, module)
        elif method == 'fastv':
            print('using fastv')
            from transformers.models.qwen2.modeling_qwen2 import Qwen2Model as _Q2Model
            target_layer_idx = getattr(args, 'target_layer_idx', 2)
            for name, module in model.named_modules():
                if _attn_class and isinstance(module, _attn_class):
                    module.forward = types.MethodType(qwen_attention_forward_fastv, module)
                    module.target_layer_idx = target_layer_idx
                if isinstance(module, _Q2Model):
                    module.forward = types.MethodType(qwen_model_forward_fastv, module)
                    module.target_layer_idx = target_layer_idx
                    module.budgets = getattr(args, 'budgets', 1.0)
                    module.origin = getattr(args, 'origin', False)

    elif method == 'visionzip':
        print('using visionzip')
        for idx, layer in enumerate(model.vision_model.encoder.layers):
            layer.attn.layer_idx = idx
            layer.attn.forward = types.MethodType(internvl_attention_forward_4B_visionzip, layer.attn)
            layer.attn._naive_attn = types.MethodType(internvl_naive_attn_4B_visionzip, layer.attn)
        model.generate = types.MethodType(internvl_generate_4B_visionzip, model)
        model.extract_feature = types.MethodType(internvl_extract_feature_4B_visionzip, model)
        model.budgets = getattr(args, 'budgets', 1.0)

    elif method == 'prumerge+':
        print('using prumerge+')
        for idx, layer in enumerate(model.vision_model.encoder.layers):
            layer.attn.layer_idx = idx
            layer.attn.forward = types.MethodType(internvl_attention_forward_4B_prumerge_plus, layer.attn)
            layer.attn._naive_attn = types.MethodType(internvl_naive_attn_4B_prumerge_plus, layer.attn)
        model.generate = types.MethodType(internvl_generate_4B_prumerge_plus, model)
        model.extract_feature = types.MethodType(internvl_extract_feature_4B_prumerge_plus, model)
        model.budgets = getattr(args, 'budgets', None)

    else:
        raise ValueError(f"Unknown method '{method}' for replace_internvl3")

    