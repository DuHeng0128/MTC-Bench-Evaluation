import argparse
import gc
import inspect
import re
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.imports import optional_import
from lmms_eval.models.model_utils.load_video import read_video_pyav_base64

process_vision_info, _has_qwen_vl = optional_import("qwen_vl_utils", "process_vision_info")
if not _has_qwen_vl:
    eval_logger.warning("Failed to import qwen_vl_utils; Please install it via `pip install qwen-vl-utils`")


@register_model("llava_onevision1_5_with_kvcache")
class Llava_OneVision1_5_with_kvcache(lmms):
    """
    Llava_OneVision1_5 Model with KV Cache Compression.
    https://huggingface.co/lmms-lab/LLaVA-OneVision-1.5-8B-Instruct
    """

    def __init__(
        self,
        pretrained: str = "lmms-lab/LLaVA-OneVision-1.5-8B-Instruct",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache: bool = True,
        attn_implementation: Optional[str] = None,
        min_pixels: int = 4 * 28 * 28,
        max_pixels: int = 700 * 28 * 28,
        max_num_frames: int = 32,
        use_custom_video_loader: Optional[bool] = False,
        fps: Optional[float] = 1.0,
        min_frames: int = 4,
        max_frames: int = 32,
        max_image_size: Optional[int] = None,
        system_prompt: Optional[str] = "You are a helpful assistant.",
        interleave_visuals: Optional[bool] = False,
        image_first: Optional[bool] = True,
        reasoning_prompt: Optional[str] = None,
        max_length: int = 2048,
        method: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__()

        defined_params = {
            "pretrained", "device", "device_map", "batch_size", "use_cache",
            "attn_implementation", "min_pixels", "max_pixels", "max_num_frames",
            "use_custom_video_loader", "fps", "min_frames", "max_frames", "max_image_size", "system_prompt",
            "interleave_visuals", "image_first", "reasoning_prompt", "max_length",
            "method",
        }
        extra_kwargs = {k: v for k, v in kwargs.items() if k not in defined_params}
        self.args = argparse.Namespace(**extra_kwargs)

        revision = kwargs.pop("revision", None)

        valid_attn_implementations = [None, "flash_attention_2", "sdpa", "eager"]
        if attn_implementation not in valid_attn_implementations:
            raise ValueError(f"attn_implementation must be one of {valid_attn_implementations}, got {attn_implementation}")

        self.use_custom_video_loader = use_custom_video_loader
        self.fps = fps
        self.max_image_size = max_image_size
        if self.max_image_size and not self.use_custom_video_loader:
            raise ValueError("max_image_size is only applicable if use_custom_video_loader is True")

        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        else:
            self._device = torch.device(device)
            self.device_map = device_map if device_map else device

        model_kwargs = {
            "torch_dtype": "auto",
            "device_map": self.device_map,
            "trust_remote_code": True,
        }
        if revision is not None:
            model_kwargs["revision"] = revision
        if attn_implementation is not None:
            model_kwargs["attn_implementation"] = attn_implementation

        # Patch HF dynamic module cache: transformers 4.57.0 no longer re-exports
        # flash_attn_varlen_func from modeling_flash_attention_utils; the cached
        # modeling_llavaonevision1_5.py may still import it from there.
        try:
            import transformers.modeling_flash_attention_utils as _fa_utils
            if not hasattr(_fa_utils, "flash_attn_varlen_func") or _fa_utils.flash_attn_varlen_func is None:
                from flash_attn import flash_attn_varlen_func as _varlen_fn
                _fa_utils.flash_attn_varlen_func = _varlen_fn
        except Exception:
            pass

        self._model = AutoModelForCausalLM.from_pretrained(pretrained, **model_kwargs).eval()

        self.method = method
        if self.method is not None:
            model_type = getattr(self._model.config, "model_type", "")
            llm_config = getattr(self._model.config, "text_config", None)
            llm_type = getattr(llm_config, "model_type", "") if llm_config else ""

            qwen2_attention_count = sum(
                1 for _, module in self._model.named_modules() if isinstance(module, Qwen2Attention)
            )

            qwen25_attention_count = 0
            try:
                from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLAttention

                qwen25_attention_count = sum(
                    1
                    for _, module in self._model.named_modules()
                    if isinstance(module, Qwen2_5_VLAttention)
                )
            except Exception:
                qwen25_attention_count = 0

            from kv_cache_compression.llava_onevision1_5_model import (
                get_llava_onevision1_5_remote_family_counts,
                is_llava_onevision1_5_remote_model,
            )
            from kv_cache_compression.monkeypatch import (
                replace_llava_onevision1_5,
                replace_qwen,
                replace_qwen2_5vl,
            )

            remote_family_counts = get_llava_onevision1_5_remote_family_counts(self._model)
            route = None
            patch_stats = None
            if is_llava_onevision1_5_remote_model(self._model):
                route = "replace_llava_onevision1_5"
                patch_stats = replace_llava_onevision1_5(
                    self.args,
                    self._model,
                    self.method.lower(),
                )
            elif qwen2_attention_count > 0 and qwen25_attention_count == 0:
                route = "replace_qwen"
                patch_stats = replace_qwen(self.args, self._model, self.method.lower())
            elif qwen25_attention_count > 0:
                route = "replace_qwen2_5vl"
                patch_stats = replace_qwen2_5vl(self.args, self._model, self.method.lower())
            else:
                raise RuntimeError(
                    "[llava_onevision1_5_with_kvcache] No supported attention modules found: "
                    f"remote_text_attention={remote_family_counts['text_attention']}, "
                    f"remote_vision_attention={remote_family_counts['vision_attention']}, "
                    f"Qwen2Attention={qwen2_attention_count}, "
                    f"Qwen2_5_VLAttention={qwen25_attention_count}; "
                    "cannot apply kvcache monkeypatch safely."
                )

            eval_logger.info(
                "[llava_onevision1_5_with_kvcache] "
                f"method={self.method.lower()} route={route} "
                f"patched_attention_count={patch_stats['patched_attention_count']} "
                f"patched_model_count={patch_stats['patched_model_count']} "
                f"patched_generate_count={patch_stats['patched_generate_count']} "
                f"remote_text_attention={remote_family_counts['text_attention']} "
                f"remote_text_model={remote_family_counts['text_model']} "
                f"remote_vision_attention={remote_family_counts['vision_attention']} "
                f"remote_vision_model={remote_family_counts['vision_model']} "
                f"(model_type='{model_type}', text_config.model_type='{llm_type}')"
            )

        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.max_num_frames = max_num_frames
        self.min_frames = min_frames
        self.max_frames = max_frames
        self.image_first = image_first
        if reasoning_prompt:
            self.reasoning_prompt = reasoning_prompt.replace("\\n", "\n")
        else:
            self.reasoning_prompt = None
        self.processor = AutoProcessor.from_pretrained(
            pretrained,
            max_pixels=max_pixels,
            min_pixels=min_pixels,
            trust_remote_code=True,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained, trust_remote_code=True)
        self.system_prompt = system_prompt
        self.interleave_visuals = interleave_visuals

        self._config = self.model.config
        self._max_length = int(max_length)
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
            ], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            if accelerator.distributed_type == DistributedType.FSDP:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self._rank = 0
            self._world_size = 1

    @property
    def config(self):
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood is not implemented for Llava_OneVision1_5_with_kvcache")

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def _get_custom_video_frame_budget(self) -> int:
        return min(self.max_frames, self.max_num_frames)

    def _get_frame_indices(self, total_frames: int) -> Optional[np.ndarray]:
        if self.max_num_frames < 1 or total_frames <= self.max_num_frames:
            return None

        capped_frames = min(total_frames, self.max_num_frames)
        if capped_frames <= 1:
            return np.array([total_frames - 1], dtype=int)

        frame_indices = np.linspace(0, total_frames - 1, capped_frames, dtype=int)
        frame_indices[-1] = total_frames - 1
        return np.unique(frame_indices)

    def _cap_video_inputs(
        self, video_inputs: Optional[List[Union[torch.Tensor, object]]]
    ) -> Optional[List[Union[torch.Tensor, object]]]:
        if video_inputs is None:
            return None

        for index, video_tensor in enumerate(video_inputs):
            if not isinstance(video_tensor, torch.Tensor):
                continue

            frame_indices = self._get_frame_indices(video_tensor.shape[0])
            if frame_indices is not None:
                video_inputs[index] = video_tensor[frame_indices]

        return video_inputs

    def _load_custom_video_content(self, video_path: str) -> Optional[List[str]]:
        try:
            return read_video_pyav_base64(
                video_path,
                num_frm=self._get_custom_video_frame_budget(),
                fps=self.fps,
                img_format="JPEG",
                max_image_size=self.max_image_size,
                return_data_urls=True,
            )
        except Exception as error:
            eval_logger.error(
                f"Failed to load video: {video_path}. Error: {error}"
            )
            return None

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        for chunk in chunks:
            batched_messages = None
            texts = None
            image_inputs = None
            video_inputs = None
            inputs = None
            filtered_inputs = None
            gen_args = None
            cont = None
            generated_ids_trimmed = None
            answers = None
            processed_visuals = None
            video_content = None
            visual_list = None
            contexts = None
            current_gen_kwargs = None
            until = None
            gen_kwargs = None

            try:
                contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
                task = task[0]
                split = split[0]
                visual_list = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]
                gen_kwargs = all_gen_kwargs[0]

                until = gen_kwargs.get("until", [self.tokenizer.decode(self.eot_token_id)])
                if isinstance(until, str):
                    until = [until]
                elif not isinstance(until, list):
                    raise ValueError(
                        "Expected `gen_kwargs['until']` to be of type "
                        f"Union[str, list], but got {type(until)}"
                    )
                until = [item for item in until if item != "\n\n"]

                if isinstance(contexts, tuple):
                    contexts = list(contexts)

                for index, context in enumerate(contexts):
                    if "<image>" in context:
                        contexts[index] = context.replace("<image>", "")

                batched_messages = []
                for index, context in enumerate(contexts):
                    if "<image>" in context:
                        context = context.replace("<image>", "")

                    message = [{"role": "system", "content": self.system_prompt}]
                    if self.reasoning_prompt:
                        context = context.strip() + self.reasoning_prompt
                        contexts[index] = context

                    processed_visuals = []
                    for visual in visual_list[index]:
                        if isinstance(visual, str) and visual.endswith(
                            (".mp4", ".avi", ".mov", ".mkv", ".webm")
                        ):
                            if self.use_custom_video_loader:
                                video_content = self._load_custom_video_content(visual)
                                if video_content is None:
                                    continue
                                processed_visuals.append(
                                    {"type": "video", "video": video_content}
                                )
                                video_content = None
                            else:
                                processed_visuals.append(
                                    {
                                        "type": "video",
                                        "video": visual,
                                        "max_pixels": self.max_pixels,
                                        "min_pixels": self.min_pixels,
                                        "fps": self.fps,
                                        "min_frames": self.min_frames,
                                        "max_frames": self.max_frames,
                                    }
                                )
                        elif isinstance(visual, Image.Image):
                            processed_visuals.append(
                                {"type": "image", "image": visual.convert("RGB")}
                            )

                    if self.interleave_visuals is False:
                        if self.image_first:
                            message.append(
                                {
                                    "role": "user",
                                    "content": processed_visuals
                                    + [{"type": "text", "text": context}],
                                }
                            )
                        else:
                            message.append(
                                {
                                    "role": "user",
                                    "content": [{"type": "text", "text": context}]
                                    + processed_visuals,
                                }
                            )
                    else:
                        image_placeholders = re.findall(r"<image \d+>", context)
                        content_parts = []
                        text_parts = re.split(r"<image \d+>", context)
                        if text_parts[0]:
                            content_parts.append(
                                {"type": "text", "text": text_parts[0]}
                            )

                        for placeholder_index, placeholder in enumerate(
                            image_placeholders
                        ):
                            image_index = (
                                int(
                                    re.search(r"<image (\d+)>", placeholder).group(1)
                                )
                                - 1
                            )
                            image_index = (
                                min(image_index, len(processed_visuals) - 1)
                                if processed_visuals
                                else 0
                            )
                            if processed_visuals and image_index < len(
                                processed_visuals
                            ):
                                content_parts.append(processed_visuals[image_index])
                            if (
                                placeholder_index + 1 < len(text_parts)
                                and text_parts[placeholder_index + 1]
                            ):
                                content_parts.append(
                                    {
                                        "type": "text",
                                        "text": text_parts[placeholder_index + 1],
                                    }
                                )

                        message.append({"role": "user", "content": content_parts})
                        del content_parts, text_parts, image_placeholders

                    batched_messages.append(message)
                    del message, processed_visuals
                    processed_visuals = None

                texts = [
                    self.processor.apply_chat_template(
                        msg, tokenize=False, add_generation_prompt=True
                    )
                    for msg in batched_messages
                ]
                image_inputs, video_inputs = process_vision_info(batched_messages)
                video_inputs = self._cap_video_inputs(video_inputs)
                inputs = self.processor(
                    text=texts,
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )

                if self.device_map == "auto":
                    inputs = inputs.to("cuda")
                else:
                    inputs = inputs.to(self.device)

                default_gen_kwargs = {
                    "max_new_tokens": 128,
                    "temperature": 0.0,
                    "top_p": None,
                    "num_beams": 1,
                }
                current_gen_kwargs = {**default_gen_kwargs, **gen_kwargs}
                pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
                do_sample = bool(
                    current_gen_kwargs.get("temperature", 0)
                    and current_gen_kwargs["temperature"] > 0
                )

                unsupported_keys = ["second_per_grid_ts"]
                filtered_inputs = {
                    key: value
                    for key, value in inputs.items()
                    if key not in unsupported_keys
                }

                gen_args = {
                    **filtered_inputs,
                    "eos_token_id": self.tokenizer.eos_token_id,
                    "pad_token_id": pad_token_id,
                    "num_beams": current_gen_kwargs["num_beams"],
                    "max_new_tokens": current_gen_kwargs["max_new_tokens"],
                    "use_cache": self.use_cache,
                }
                if do_sample:
                    top_p = current_gen_kwargs.get("top_p", 1.0)
                    gen_args.update(
                        do_sample=True,
                        temperature=float(current_gen_kwargs.get("temperature", 1.0)),
                        top_p=float(top_p) if top_p is not None else None,
                    )
                with torch.inference_mode():
                    cont = self.model.generate(**gen_args)

                generated_ids_trimmed = [
                    out_ids[len(in_ids) :]
                    for in_ids, out_ids in zip(inputs.input_ids, cont)
                ]
                answers = self.processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                for index, answer in enumerate(answers):
                    for term in until:
                        if term:
                            answer = answer.split(term)[0]
                    answers[index] = answer

                for answer, context in zip(answers, contexts):
                    res.append(answer)
                    self.cache_hook.add_partial(
                        "generate_until", (context, gen_kwargs), answer
                    )
                    pbar.update(1)
            finally:
                del batched_messages
                del texts
                del image_inputs
                del video_inputs
                del inputs
                del filtered_inputs
                del gen_args
                del cont
                del generated_ids_trimmed
                del answers
                del processed_visuals
                del video_content
                del visual_list
                del contexts
                del current_gen_kwargs
                del until
                del gen_kwargs
                del chunk
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

        res = re_ords.get_original(res)
        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")
