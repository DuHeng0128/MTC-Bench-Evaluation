import os
import re
import json
import yaml
import statistics
import sys
import time
import string
import base64
from io import BytesIO
from typing import Dict, List, Optional, Union, Tuple, Any, Counter
import cv2
import requests
from decord import VideoReader, cpu
from loguru import logger as eval_logger
from lmms_eval.tasks._task_utils.file_utils import generate_submission_file
from functools import partial
import datasets
from datasets import load_dataset
import datetime
import lmms_eval.tasks._task_utils.file_utils as file_utils
from lmms_eval.tasks._task_utils.vqa_eval_metric import EvalAIAnswerProcessor
from PIL import ImageDraw
try:
    from pycocoevalcap.eval import Bleu, Cider, COCOEvalCap, Meteor, Rouge, Spice
    _HAS_COCO_EVAL = True
except ImportError:
    _HAS_COCO_EVAL = False
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocotools.coco import COCO
from pathlib import Path
import pandas as pd
from lmms_eval.tasks.mtcbench.mathvista_evals import MathVistaEvaluator
from lmms_eval.tasks.mtcbench.hrbench_evals import HRBenchEval
from lmms_eval.tasks.mtcbench.mmbench_evals import MMBench_Evaluator
import ast
import random
import math
import numpy as np
from lmms_eval.tasks._task_utils.video_loader import get_cache_dir, get_video
from collections import Counter, defaultdict
import torch
from PIL import Image
from tqdm import tqdm

os.environ["HF_HOME"] = "/root/MTC-Bench/hf_cache"
GQA_RAW_IMAGE_DATASET = None
GQA_ID2IMAGE = None
COCO_METRICS = ["Bleu_4", "Bleu_3", "Bleu_2", "Bleu_1", "METEOR", "ROUGE_L", "CIDEr"] # , "SPICE"]

from openai import OpenAI
API_TYPE = os.getenv("API_TYPE", "openai")
if API_TYPE == "openai":
    API_URL = os.getenv("OPENAI_API_URL", "YOUR_OPENAI_API_URL")
    API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY")
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
elif API_TYPE == "azure":
    API_URL = os.getenv("AZURE_ENDPOINT", "YOUR_AZURE_ENDPOINT")
    API_KEY = os.getenv("AZURE_API_KEY", "YOUR_AZURE_API_KEY")
    headers = {
        "api-key": API_KEY,
        "Content-Type": "application/json",
    }
client = OpenAI(base_url=API_URL, api_key=API_KEY, timeout=30.0)
DEFAULT_GPT_EVAL_MODEL = "gpt-3.5-turbo-1106"


def _get_gpt_eval_model(task_config: Dict[str, Any]) -> str:
    metadata = task_config.get("metadata", {}) if isinstance(task_config, dict) else {}
    model_name = metadata.get("gpt_eval_model_name") if isinstance(metadata, dict) else None
    if isinstance(model_name, str) and model_name.strip():
        return model_name.strip()
    return DEFAULT_GPT_EVAL_MODEL


with open(Path(__file__).parent / "gqa_lite.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)

    gpa_config = yaml.safe_load("".join(safe_data))

def gqa_doc_to_visual(doc):
    global GQA_RAW_IMAGE_DATASET
    global GQA_ID2IMAGE
    if GQA_RAW_IMAGE_DATASET is None:
        image_dataset_path = gpa_config["dataset_path"] + "/testdev_balanced_images"
        GQA_RAW_IMAGE_DATASET = load_dataset(image_dataset_path, split="train", token=True)
        GQA_ID2IMAGE = {}
        for row in GQA_RAW_IMAGE_DATASET:
            GQA_ID2IMAGE[row["id"]] = row["image"].convert("RGB")
    image = GQA_ID2IMAGE[doc["imageId"]]
    return [image]

def gqa_doc_to_text(doc, lmms_eval_specific_kwargs):
    question = doc["question"]
    pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    post_prompt = lmms_eval_specific_kwargs["post_prompt"]
    return f"{pre_prompt}{question}{post_prompt}"


def vqav2_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]

def vqav2_process_results(doc, result):
    eval_ai_processor = EvalAIAnswerProcessor()
    assert len(result) == 1, f"The result should be a list of length 1, but got {len(result)}."
    resAns = eval_ai_processor(result[0])
    accuracy = 0

    if "answers" in doc and doc["answers"] is not None:
        for ansDic in doc["answers"]:
            ansDic["answer"] = ansDic["answer"].replace("\n", " ")
            ansDic["answer"] = ansDic["answer"].replace("\t", " ")
            ansDic["answer"] = ansDic["answer"].strip()
        gtAcc = []
        gtAnswers = [ans["answer"] for ans in doc["answers"]]

        if len(set(gtAnswers)) > 1:
            for ansDic in doc["answers"]:
                ansDic["answer"] = eval_ai_processor.process_punctuation(ansDic["answer"])
                ansDic["answer"] = eval_ai_processor.process_digit_article(ansDic["answer"])
            resAns = eval_ai_processor.process_punctuation(resAns)
            resAns = eval_ai_processor.process_digit_article(resAns)

        for gtAnsDatum in doc["answers"]:
            otherGTAns = [item for item in doc["answers"] if item != gtAnsDatum]
            matchingAns = [item for item in otherGTAns if item["answer"] == resAns]
            acc = min(1, float(len(matchingAns)) / 3)
            gtAcc.append(acc)
        accuracy = statistics.mean(gtAcc)

    return {
        "exact_match": accuracy,
        "submission": {
            "question_id": doc["question_id"],
            "answer": resAns,
        },
    }

def vqav2_process_results_test(doc, result):
    res = vqav2_process_results(doc, result)
    return {
        "submission": res["submission"],
    }

def vqav2_process_results_val(doc, result):
    res = vqav2_process_results(doc, result)
    return {
        "exact_match": res["exact_match"],
    }

def vqav2_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    pre_prompt = ""
    post_prompt = ""
    if "pre_prompt" in lmms_eval_specific_kwargs:
        pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    if "post_prompt" in lmms_eval_specific_kwargs:
        post_prompt = lmms_eval_specific_kwargs["post_prompt"]
    return f"{pre_prompt}{doc['question']}{post_prompt}"

def vqav2_aggregate_submissions(results, args):
    now_date_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    submission_file_name = f"vqav2-test-submission-{now_date_time}.json"
    args.output_path = args.output_path if args.output_path else "./"
    path = file_utils.generate_submission_file(submission_file_name, args)
    with open(path, "w") as f:
        json.dump(results, f)
    eval_logger.info(f"Submission file saved to {path}")


def vizwiz_vqa_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]

def vizwiz_vqa_process_results(doc, result):
    eval_ai_processor = EvalAIAnswerProcessor()
    assert len(result) == 1, f"The result should be a list of length 1, but got {len(result)}."
    resAns = eval_ai_processor(result[0])
    accuracy = 0

    if "answers" in doc and doc["answers"] is not None:
        gtAcc = []

        for i in range(len(doc["answers"])):
            doc["answers"][i] = eval_ai_processor(doc["answers"][i])

        for i in range(len(doc["answers"])):
            otherGTAns = [doc["answers"][j] for j in range(len(doc["answers"])) if i != j]
            matchingAns = [item for item in otherGTAns if item == resAns]
            acc = min(1, float(len(matchingAns)) / 3)
            gtAcc.append(acc)
        if gtAcc:
            accuracy = statistics.mean(gtAcc)
        else:
            accuracy = 0

    return {
        "exact_match": accuracy,
        "submission": {
            "image": f"{doc['question_id']}.jpg",
            "answer": resAns,
        },
    }

def vizwiz_vqa_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    pre_prompt = ""
    post_prompt = ""
    if "pre_prompt" in lmms_eval_specific_kwargs:
        pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    if "post_prompt" in lmms_eval_specific_kwargs:
        post_prompt = lmms_eval_specific_kwargs["post_prompt"]
    text = f"{pre_prompt}{doc['question'].capitalize()}{post_prompt}"
    return text

def vizwiz_vqa_aggregate_submissions(results, args):
    now_date_time = datetime.datetime.now().strftime("%Y-%m%d-%H%M-%S")
    submission_file_name = f"vizwiz_vqa-test-submission-{now_date_time}.json"
    args.output_path = args.output_path if args.output_path else "./"
    path = generate_submission_file(submission_file_name, args)
    with open(path, "w") as f:
        json.dump(results, f)
    print(f"Submission file saved to {path}")


def textvqa_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]

def textvqa_process_results(doc, result):
    eval_ai_processor = EvalAIAnswerProcessor()
    assert len(result) == 1, f"The result should be a list of length 1, but got {len(result)}."
    resAns = eval_ai_processor(result[0])
    accuracy = 0

    if "answers" in doc and doc["answers"] is not None:
        gtAcc = []

        for i in range(len(doc["answers"])):
            doc["answers"][i] = eval_ai_processor(doc["answers"][i])

        for i in range(len(doc["answers"])):
            otherGTAns = [doc["answers"][j] for j in range(len(doc["answers"])) if i != j]
            matchingAns = [item for item in otherGTAns if item == resAns]
            acc = min(1, float(len(matchingAns)) / 3)
            gtAcc.append(acc)
        accuracy = statistics.mean(gtAcc)

    return {
        "exact_match": accuracy,
        "submission": {
            "question_id": doc["question_id"],
            "answer": resAns,
        },
    }

def textvqa_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    pre_prompt = ""
    post_post = ""
    ocr_ref = ""
    if lmms_eval_specific_kwargs:
        if "pre_prompt" in lmms_eval_specific_kwargs:
            pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
        if "post_prompt" in lmms_eval_specific_kwargs:
            post_prompt = lmms_eval_specific_kwargs["post_prompt"]
        if "ocr" in lmms_eval_specific_kwargs and lmms_eval_specific_kwargs["ocr"]:
            ocr_ref = f"\nReference OCR token: {', '.join(doc['ocr_tokens'])}"
    return f"{pre_prompt}{doc['question'].capitalize()}{ocr_ref}{post_prompt}"

def textvqa_aggregate_submissions(results, args):
    now_date_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    args.output_path = args.output_path if args.output_path else "./"
    path = generate_submission_file(f"textvqa_submission_{now_date_time}.json", args)
    with open(path, "w") as f:
        json.dump(results, f)
    # print(f"Submission file saved to {path}")
    eval_logger.info(f"Submission file saved to {path}")


def docvqa_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]

def docvqa_doc_to_text(doc, lmms_eval_specific_kwargs):
    question = doc["question"]
    pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    post_prompt = lmms_eval_specific_kwargs["post_prompt"]
    return f"{pre_prompt}{question}{post_prompt}"

def docvqa_test_process_results(doc, results):
    pred = results[0]
    questionId = doc["questionId"]
    return {"anls": {"questionId": int(questionId), "answer": pred}, "submission": {"questionId": int(questionId), "answer": pred}}

def docvqa_test_aggregate_results(results, args):
    # save results as json
    args.output_path = args.output_path if args.output_path else "./"
    path = generate_submission_file("docvqa_test_for_submission.json", args)
    with open(path, "w") as f:
        json.dump(results, f)
    eval_logger.info(f"Results saved to {path}")


def infovqa_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]

def infovqa_doc_to_text(doc, lmms_eval_specific_kwargs):
    question = doc["question"]
    pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    post_prompt = lmms_eval_specific_kwargs["post_prompt"]
    return f"{pre_prompt}{question}{post_prompt}"

def infovqa_test_process_results(doc, results):
    pred = results[0]
    questionId = doc["questionId"]
    return {"submission": {"questionId": int(questionId), "answer": pred}}

def infovqa_test_aggregate_results(results, args):
    # save results as json
    args.output_path = args.output_path if args.output_path else "./"
    file = generate_submission_file("infovqa_test_for_submission.json", args)
    with open(file, "w") as f:
        json.dump(results, f)
    eval_logger.info(f"Results saved to {file}")


def chartqa_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]

def chartqa_doc_to_text(doc, lmms_eval_specific_kwargs):
    question = doc["question"]
    pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    post_prompt = lmms_eval_specific_kwargs["post_prompt"]
    return f"{pre_prompt}{question}{post_prompt}"

def chartqa_process_results(doc, results):
    pred = results[0]
    type = doc["type"]
    score = relaxed_correctness(pred, doc["answer"])
    score = 1.0 if score else 0.0
    return_dict = {"relaxed_overall": score}
    if type == "human_test":
        return_dict["relaxed_human_split"] = score
    else:
        return_dict["relaxed_augmented_split"] = score
    return return_dict

def relaxed_correctness(prediction, target, max_relative_change: float = 0.05) -> bool:

    def _to_float(text: str):
        try:
            if text.endswith("%"):
                return float(text.rstrip("%")) / 100.0
            else:
                return float(text)
        except ValueError:
            return None

    prediction_float = _to_float(prediction)
    target_float = _to_float(target)
    if prediction_float is not None and target_float:
        relative_change = abs(prediction_float - target_float) / abs(target_float)
        return relative_change <= max_relative_change
    else:
        return prediction.lower() == target.lower()


OCRBench_score = {
    "Regular Text Recognition": 0,
    "Irregular Text Recognition": 0,
    "Artistic Text Recognition": 0,
    "Handwriting Recognition": 0,
    "Digit String Recognition": 0,
    "Non-Semantic Text Recognition": 0,
    "Scene Text-centric VQA": 0,
    "Doc-oriented VQA": 0,
    "Key Information Extraction": 0,
    "Handwritten Mathematical Expression Recognition": 0,
}
# math QA en, 

def ocrbench_doc_to_visual(doc):
    image_bytes = doc["image"]["bytes"]
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    return [image]

def ocrbench_doc_to_text(doc):
    question = doc["question"].strip()
    return f"{question}"

def ocrbench_process_results(doc, results):
    pred = results[0].lower().strip()
    gt_ans = doc["answers"]
    dataset_name = doc["dataset_name"]

    score = 0
    if dataset_name == "HME100k":
        if type(gt_ans) == list:
            for j in range(len(gt_ans)):
                answer = gt_ans[j].strip().replace("\n", " ").replace(" ", "")
                predict = pred.strip().replace("\n", " ").replace(" ", "")
                if answer in predict:
                    score = 1
        else:
            answer = gt_ans.strip().replace("\n", " ").replace(" ", "")
            predict = pred.strip().replace("\n", " ").replace(" ", "")
            if answer in predict:
                score = 1
    else:
        if type(gt_ans) == list:
            for j in range(len(gt_ans)):
                answer = gt_ans[j].lower().strip().replace("\n", " ")
                predict = pred.lower().strip().replace("\n", " ")
                if answer in predict:
                    score = 1
        else:
            answer = gt_ans.lower().strip().replace("\n", " ")
            predict = pred.lower().strip().replace("\n", " ")
            if answer in predict:
                score = 1
    return {
        "ocrbench_accuracy": {"question_type": doc["type"], "score": score, "prediction": pred, "ground_truth": gt_ans},
    }


def ocrbench_aggregate_accuracy(results, args):
    Final_score = 0
    length = 0
    for result in results:
        Final_score += result["score"]
        length += 1
    # recognition_score = (
    #     OCRBench_score["Regular Text Recognition"]
    #     + OCRBench_score["Irregular Text Recognition"]
    #     + OCRBench_score["Artistic Text Recognition"]
    #     + OCRBench_score["Handwriting Recognition"]
    #     + OCRBench_score["Digit String Recognition"]
    #     + OCRBench_score["Non-Semantic Text Recognition"]
    # )
    # Final_score = recognition_score + OCRBench_score["Scene Text-centric VQA"] + OCRBench_score["Doc-oriented VQA"] + OCRBench_score["Key Information Extraction"] + OCRBench_score["Handwritten Mathematical Expression Recognition"]
    # args.output_path = args.output_path if args.output_path else "./"
    # file_name = generate_submission_file("ocrbench_results.txt", args, subpath="results")
    # with open(file_name, "w") as f:
    #     print("######################### OCRBench #############################", file=f)
    #     print(f"Text Recognition(Total 300): {recognition_score}", file=f)
    #     print("---------------- Details of Recognition Score ------------------", file=f)
    #     print(f"Regular Text Recognition(Total 50): {OCRBench_score['Regular Text Recognition']}", file=f)
    #     print(f"Irregular Text Recognition(Total 50): {OCRBench_score['Irregular Text Recognition']}", file=f)
    #     print(f"Artistic Text Recognition(Total 50): {OCRBench_score['Artistic Text Recognition']}", file=f)
    #     print(f"Handwriting Recognition(Total 50): {OCRBench_score['Handwriting Recognition']}", file=f)
    #     print(f"Digit String Recognition(Total 50): {OCRBench_score['Digit String Recognition']}", file=f)
    #     print(f"Non-Semantic Text Recognition(Total 50): {OCRBench_score['Non-Semantic Text Recognition']}", file=f)
    #     print("----------------------------------------------------------------", file=f)
    #     print(f"Scene Text-centric VQA(Total 200): {OCRBench_score['Scene Text-centric VQA']}", file=f)
    #     print("----------------------------------------------------------------", file=f)
    #     print(f"Doc-oriented VQA(Total 200): {OCRBench_score['Doc-oriented VQA']}", file=f)
    #     print("----------------------------------------------------------------", file=f)
    #     print(f"Key Information Extraction(Total 200): {OCRBench_score['Key Information Extraction']}", file=f)
    #     print("----------------------------------------------------------------")
    #     print(f"Handwritten Mathematical Expression Recognition(Total 100): {OCRBench_score['Handwritten Mathematical Expression Recognition']}", file=f)
    #     print("--------------------- Final Score ------------------------------", file=f)
    #     print(f"Final Score(Total 1000): {Final_score}", file=f)
    # eval_logger.info(f"OCR Bench results saved to {file_name}")
    # return {"Final Score":Final_score,"Text Recognition":recognition_score,'Scene Text-centric VQA':OCRBench_score['Scene Text-centric VQA'],'Doc-oriented VQA':OCRBench_score['Doc-oriented VQA'],'Key Information Extraction':OCRBench_score['Key Information Extraction'],'Handwritten Mathematical Expression Recognition':OCRBench_score['Handwritten Mathematical Expression Recognition']}
    return Final_score / length  # return the final score as accuracy


def refcoco_bbox_doc_to_visual(doc):
    bbox = doc["bbox"]
    image = doc["image"]
    draw = ImageDraw.Draw(image)
    # Origin format (top x, top y, width, height)
    bbox_xy = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
    draw.rectangle(bbox_xy, outline="red")
    return [image.convert("RGB")]

def refcoco_seg_doc_to_visual(doc):
    seg = doc["segmentation"]
    # image_bytes = doc["image"]["bytes"]
    # image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image = doc["image"]
    draw = ImageDraw.Draw(image)
    draw.polygon(seg)
    return [image.convert("RGB")]

def refcoco_doc_to_text(doc):
    # question = doc["question"]
    # return f"{question}\nAnswer the question using a single word or phrase."
    return "Provide a short description for this region."

def refcoco_process_result(doc, result):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name (in this case coco_bleu), value: metric value
    """
    pred = result[0] if len(result) > 0 else ""
    ann_id = doc["question_id"]
    data_dict = {"answer": doc["answer"], "pred": pred, "ann_id": ann_id}
    return {f"refcoco_{metric}": data_dict for metric in COCO_METRICS}

def refcoco_aggregation_result(results, metric):
    scorers = [(Bleu(4), "Bleu_1"), (Bleu(4), "Bleu_2"), (Bleu(4), "Bleu_3"), (Bleu(4), "Bleu_4"), (Meteor(), "METEOR"), (Rouge(), "ROUGE_L"), (Cider(), "CIDEr")]  # , (Spice(), "SPICE")]
    scorers_dict = {s[1]: s for s in scorers}

    stored_results = []
    # In order to make the coco eval tools to successfully create index
    # We need at least two dict in the dataset
    # 'annotation' and 'images'
    # 'annotation' exactly reproduce the original annotation
    # 'images' however only need the image id which is contained in the file name
    dataset = {"annotations": [], "images": []}
    idx = 0
    ann_id = 0
    for result in results:
        stored_results.append({"image_id": idx, "caption": result["pred"]})
        for s in result["answer"]:
            dataset["annotations"].append({"image_id": idx, "caption": s, "id": ann_id})
            ann_id += 1

        dataset["images"].append({"id": idx})
        idx += 1

    coco = COCO()
    # Manually create index here
    coco.dataset = dataset
    coco.createIndex()

    coco_result = coco.loadRes(stored_results)
    coco_eval = COCOEvalCap(coco, coco_result)

    imgIds = coco_eval.params["image_id"]
    gts = {}
    res = {}
    for imgId in imgIds:
        gts[imgId] = coco_eval.coco.imgToAnns[imgId]
        res[imgId] = coco_eval.cocoRes.imgToAnns[imgId]

    eval_logger.info("tokenization...")
    tokenizer = PTBTokenizer()
    gts = tokenizer.tokenize(gts)
    res = tokenizer.tokenize(res)

    eval_logger.info(f"Computing {metric} scores...")

    score, scores = scorers_dict[metric][0].compute_score(gts, res)
    # coco_eval.setEval(score, metric)

    # When metric is one of the Bleu, score will be a list
    if type(score) == list:
        n = int(metric.split("_")[-1])
        score = score[n - 1]

    return score

def refcoco_bleu4(results):
    return refcoco_aggregation_result(results, "Bleu_4")

def refcoco_bleu3(results):
    return refcoco_aggregation_result(results, "Bleu_3")

def refcoco_bleu2(results):
    return refcoco_aggregation_result(results, "Bleu_2")

def refcoco_bleu1(results):
    return refcoco_aggregation_result(results, "Bleu_1")

def refcoco_meteor(results):
    return refcoco_aggregation_result(results, "METEOR")

def refcoco_rougel(results):
    return refcoco_aggregation_result(results, "ROUGE_L")

def refcoco_cider(results):
    return refcoco_aggregation_result(results, "CIDEr")

def refcoco_spice(results):
    return refcoco_aggregation_result(results, "SPICE")


with open(Path(__file__).parent / "mathvista_test.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)

    mathvista_config = yaml.safe_load("".join(safe_data))

mathvista_evaluator = MathVistaEvaluator(api_key=API_KEY, gpt_model=_get_gpt_eval_model(mathvista_config))

def mathvista_doc_to_visual(doc):
    image_bytes = doc["decoded_image"]["bytes"]
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    return [image]


def mathvista_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    problem = {
        "question_type": doc["question_type"],
        "answer_type": doc["answer_type"],
        "question": doc["question"],
        "unit": doc["unit"] if "unit" in doc else "",
        "caption": doc["caption"] if "caption" in doc else "",
        "ocr": doc["ocr"] if "ocr" in doc else "",
        "choices": doc["choices"],
        "answer": doc["answer"] if "answer" in doc else None,
        "precision": doc["precision"] if "precision" in doc else 0,
    }
    query_prompt = mathvista_evaluator.create_one_query(
        problem,
        shot_num=lmms_eval_specific_kwargs["shot"],
        shot_type=lmms_eval_specific_kwargs["shot_type"],
        use_caption=lmms_eval_specific_kwargs["use_caption"],
        use_ocr=lmms_eval_specific_kwargs["use_ocr"],
    )
    return query_prompt

def mathvista_process_results(doc, results):
    prediction = results[0].strip()
    problem = {
        "question_type": doc["question_type"],
        "answer_type": doc["answer_type"],
        "query": doc["query"],
        "choices": doc["choices"],
        "answer": doc["answer"] if "answer" in doc else None,
        "precision": doc["precision"] if "precision" in doc else 0,
    }
    extraction = mathvista_evaluator.extract_answer(prediction, problem, mathvista_config["metadata"]["quick_extract"])

    prediction = mathvista_evaluator.normalize_extracted_answer(extraction, problem["choices"], problem["question_type"], problem["answer_type"], problem["precision"])
    # set test set answer to None
    true_false = mathvista_evaluator.safe_equal(prediction, problem["answer"]) if problem["answer"] is not None else False

    result = {
        "question_id": doc["pid"],
        "extraction": extraction,
        "prediction": prediction,
        "true_false": true_false,
        "question_type": doc["question_type"],
        "answer_type": doc["answer_type"],
        "precision": doc["precision"] if "precision" in doc else 0,
        "metadata": doc["metadata"],
    }

    return {
        "gpt_eval_score": result,
        "submission": result,
    }

def mathvista_aggregate_results(results, args, *, calculate_gain=False, random_scores=None):
    split_flag = results[0]["metadata"]["split"]
    full_pids = [result["question_id"] for result in results]
    total = len(results)
    correct = sum(1 for idx, pid in enumerate(full_pids) if results[idx]["true_false"])
    accuracy = round(correct / total * 100, 2)
    scores = {"average": {"accuracy": accuracy, "correct": correct, "total": total}}

    for result in results:
        result.update(result.pop("metadata"))

    results_dict = {result["question_id"]: result for result in results}
    df = pd.DataFrame(results_dict).T
    target_keys = ["question_type", "answer_type", "language", "source", "category", "task", "context", "grade", "skills"]

    for key in target_keys:
        values = df[key].explode().unique() if key == "skills" else df[key].unique()
        scores[key] = {}
        for value in values:
            correct, total, acc = mathvista_evaluator.get_acc_with_contion(df, key, value)
            if total > 0:
                scores[key][value] = {"accuracy": acc, "correct": correct, "total": total}
        scores[key] = dict(sorted(scores[key].items(), key=lambda item: float(item[1]["accuracy"]), reverse=True))

    if calculate_gain:
        for key in scores:
            if key == "average":
                gain = round(float(scores[key]["accuracy"]) - float(random_scores[key]["accuracy"]), 2)
                scores[key]["acc_gain"] = gain
            else:
                for sub_key in scores[key]:
                    gain = round(float(scores[key][sub_key]["accuracy"]) - float(random_scores[key][sub_key]["accuracy"]), 2)
                    scores[key][sub_key]["acc_gain"] = gain

    args.output_path = args.output_path if args.output_path else "./"
    path = generate_submission_file(f"mathvista_{split_flag}_scores.json", args)
    with open(path, "w") as f:
        json.dump(results_dict, f, indent=4)
    eval_logger.info(f"Saved results to {path}")
    if scores["average"]["accuracy"] == 0:
        return None
    return scores["average"]["accuracy"]


with open(Path(__file__).parent / "mmmu_test.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)

    mmmu_config = yaml.safe_load("".join(safe_data))
MULTI_CHOICE_PROMPT = "Answer with the option's letter from the given choices directly."
OPEN_ENDED_PROMPT = "Answer the question using a single word or phrase."

def replace_images_tokens(input_string):
    for i in range(1, 8):
        question_text = f"<image {i}>"
        query_text = "<image>"
        if question_text in input_string:
            input_string = input_string.replace(question_text, query_text)
    return input_string


def parse_options(options):
    option_letters = [chr(ord("A") + i) for i in range(len(options))]
    choices_str = "\n".join([f"{option_letter}. {option}" for option_letter, option in zip(option_letters, options)])
    return choices_str

def construct_prompt(doc):
    question = doc["question"]
    if doc["question_type"] == "multiple-choice":
        # Weirdly, data["options"] is a string in MMMU Huggingface dataset
        parsed_options = parse_options(ast.literal_eval(doc["options"]))
        # parsed_options already prepends a newline so no need to add space here
        question = f"{question}\n{parsed_options}\n\n{MULTI_CHOICE_PROMPT}"
    else:
        question = f"{question}\n\n{OPEN_ENDED_PROMPT}"
    return question


def mmmu_doc_to_text(doc):
    question = construct_prompt(doc)
    if mmmu_config["metadata"]["interleaved_format"]:
        question = replace_images_tokens(question)
    return question


def mmmu_doc_to_visual(doc):
    prompt = construct_prompt(doc)
    image_tokens = re.findall(r"<image \d+>", prompt)
    # Remove <> and  swap space as _
    image_tokens = sorted(list(set([image_token.strip("<>").replace(" ", "_") for image_token in image_tokens])))
    visual = []
    for image_token in image_tokens:
        image_bytes = doc[image_token]["bytes"]
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        visual.append(image)
    return visual


def mmmu_process_results(doc, results):
    pred = results[0]
    if doc["question_type"] == "multiple-choice":
        index2ans, all_choices = get_multi_choice_info_mmmu(ast.literal_eval(doc["options"]))
        parsed_pred = parse_multi_choice_response_mmmu(pred, all_choices, index2ans)
    else:
        parsed_pred = parse_open_response_mmmu(pred)
    id = doc["id"]
    mmmu_acc = {"id": id, "subdomain": extract_subset_name(doc["id"]), "question_type": doc["question_type"], "answer": doc["answer"], "parsed_pred": parsed_pred}
    return {
        "mmmu_acc": mmmu_acc,
        "submission": {
            id: pred,
        },
    }


def extract_subset_name(input_string):
    # Define a regex pattern to match "validation_" at the beginning and "_<number>" at the end
    split = input_string.split("_")[0]
    pattern = re.compile(rf"^{split}_(.+?)_\d+$")
    match = pattern.search(input_string)
    if match:
        return match.group(1)
    else:
        raise ValueError(f'No match found in "{input_string}"')

def mmmu_test_aggregate_results_for_submission(results, args):
    args.output_path = args.output_path if args.output_path else "./"
    path = generate_submission_file("mmmu_test_for_submission.json", args)
    results_dict = {list(item.keys())[0]: list(item.values())[0] for item in results}
    with open(path, "w") as f:
        json.dump(results_dict, f)
    eval_logger.info(f"Results saved to {path}.")


def mmmu_aggregate_results(results):
    evaluation_result = {}
    subset_to_eval_samples = defaultdict(list)
    for result in results:
        subset_to_eval_samples[result["subdomain"]].append(result)
    for subset, sub_eval_samples in subset_to_eval_samples.items():
        judge_dict, metric_dict = evaluate_mmmu(sub_eval_samples)
        metric_dict.update({"num_example": len(sub_eval_samples)})
        evaluation_result[subset] = metric_dict
    printable_results = {}
    for domain, in_domain_cats in DOMAIN_CAT2SUB_CAT.items():
        in_domain_cat_results = {}
        for cat_name in in_domain_cats:
            if cat_name in evaluation_result.keys():
                in_domain_cat_results[cat_name] = evaluation_result[cat_name]
            else:
                pass
        in_domain_ins_acc = calculate_ins_level_acc(in_domain_cat_results)
        in_domain_data_num = sum([cat_results["num_example"] for cat_results in in_domain_cat_results.values()])
        printable_results["Overall-" + domain] = {
            "num": int(in_domain_data_num),
            "acc": round(in_domain_ins_acc, 5),
        }
        # add sub category
        for cat_name, cat_results in in_domain_cat_results.items():
            printable_results[cat_name] = {
                "num": int(cat_results["num_example"]),
                "acc": round(cat_results["acc"], 5),
            }
    all_ins_acc = calculate_ins_level_acc(evaluation_result)
    printable_results["Overall"] = {
        "num": sum([cat_results["num_example"] for cat_results in evaluation_result.values()]),
        "acc": round(all_ins_acc, 5),
    }
    print(printable_results)
    return printable_results["Overall"]["acc"]


##################
# Helper functions written by official MMMU repo.
##################


def calculate_ins_level_acc(results):
    """Calculate the instruction level accuracy for given Subject results
    https://github.com/MMMU-Benchmark/MMMU/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/eval_utils.py#L246
    """
    acc = 0
    ins_num = 0
    for cat_results in results.values():
        acc += cat_results["acc"] * cat_results["num_example"]
        ins_num += cat_results["num_example"]
    if ins_num == 0:
        return 0
    return acc / ins_num

DOMAIN_CAT2SUB_CAT = {
    "Art and Design": ["Art", "Art_Theory", "Design", "Music"],
    "Business": ["Accounting", "Economics", "Finance", "Manage", "Marketing"],
    "Science": [
        "Biology",
        "Chemistry",
        "Geography",
        "Math",
        "Physics",
    ],
    "Health and Medicine": [
        "Basic_Medical_Science",
        "Clinical_Medicine",
        "Diagnostics_and_Laboratory_Medicine",
        "Pharmacy",
        "Public_Health",
    ],
    "Humanities and Social Science": [
        "History",
        "Literature",
        "Sociology",
        "Psychology",
    ],
    "Tech and Engineering": [
        "Agriculture",
        "Architecture_and_Engineering",
        "Computer_Science",
        "Electronics",
        "Energy_and_Power",
        "Materials",
        "Mechanical_Engineering",
    ],
}


def eval_multi_choice(gold_i, pred_i):
    """
    Evaluate a multiple choice instance.
    https://github.com/MMMU-Benchmark/MMMU/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/eval_utils.py#L175
    """
    correct = False
    # only they are exactly the same, we consider it as correct
    if isinstance(gold_i, list):
        for answer in gold_i:
            if answer == pred_i:
                correct = True
                break
    else:  # gold_i is a string
        if gold_i == pred_i:
            correct = True
    return correct

def eval_open(gold_i, pred_i):
    """
    Evaluate an open question instance
    https://github.com/MMMU-Benchmark/MMMU/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/eval_utils.py#L191
    """
    correct = False
    if isinstance(gold_i, list):
        # use float to avoid trivial matches
        norm_answers = []
        for answer in gold_i:
            norm_answers.extend(normalize_str(answer))
    else:
        norm_answers = normalize_str(gold_i)
    for pred in pred_i:  # pred is already normalized in parse response phase
        if isinstance(pred, str):  # if it's a string, then find if ans in the pred_i
            for norm_ans in norm_answers:
                # only see if the string answer in the string pred
                if isinstance(norm_ans, str) and norm_ans in pred:
                    if not correct:
                        correct = True
                    break
        else:  # it's a float number
            if pred in norm_answers:
                if not correct:
                    correct = True
                break
    return correct


def evaluate_mmmu(samples):
    """
    Batch evaluation for multiple choice and open questions.
    https://github.com/MMMU-Benchmark/MMMU/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/eval_utils.py#L219
    """
    pred_correct = 0
    judge_dict = dict()
    for sample in samples:
        gold_i = sample["answer"]
        pred_i = sample["parsed_pred"]
        if sample["question_type"] == "multiple-choice":
            correct = eval_multi_choice(gold_i, pred_i)
        else:  # open question
            correct = eval_open(gold_i, pred_i)

        if correct:
            judge_dict[sample["id"]] = "Correct"
            pred_correct += 1
        else:
            judge_dict[sample["id"]] = "Wrong"

    if len(samples) == 0:
        return {"acc": 0}
    return judge_dict, {"acc": pred_correct / len(samples)}


def parse_multi_choice_response_mmmu(response, all_choices, index2ans):
    """
    Parse the prediction from the generated response.
    Return the predicted index e.g., A, B, C, D.
    https://github.com/MMMU-Benchmark/MMMU/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/eval_utils.py#L10
    """
    for char in [",", ".", "!", "?", ";", ":", "'"]:
        response = response.strip(char)
    response = " " + response + " "  # add space to avoid partial match

    index_ans = True
    ans_with_brack = False
    candidates = []
    for choice in all_choices:  # e.g., (A) (B) (C) (D)
        if f"({choice})" in response:
            candidates.append(choice)
            ans_with_brack = True

    if len(candidates) == 0:
        for choice in all_choices:  # e.g., A B C D
            if f"{choice} " in response:
                candidates.append(choice)

    if len(candidates) == 0:
        for choice in all_choices:  # e.g., A. B. C. D.
            if f"{choice}." in response:
                candidates.append(choice)

    # if all above doesn't get candidates, check if the content is larger than 5 tokens and try to parse the example
    if len(candidates) == 0 and len(response.split()) > 5:
        for index, ans in index2ans.items():
            if ans.lower() in response.lower():
                candidates.append(index)
                index_ans = False  # it's content ans.

    if len(candidates) == 0:  # still not get answer, randomly choose one.
        pred_index = random.choice(all_choices)
    elif len(candidates) > 1:
        start_indexes = []
        if index_ans:
            if ans_with_brack:
                for can in candidates:
                    index = response.rfind(f"({can})")
                    start_indexes.append(index)  # -1 will be ignored anyway
                # start_indexes = [generated_response.index(f'({can})') for can in candidates]
            else:
                for can in candidates:
                    index = response.rfind(f" {can} ")
                    start_indexes.append(index)
        else:
            for can in candidates:
                index = response.lower().rfind(index2ans[can].lower())
                start_indexes.append(index)
        # get the last one
        pred_index = candidates[np.argmax(start_indexes)]
    else:  # if only one candidate, use it.
        pred_index = candidates[0]

    return pred_index


def extract_numbers(string):
    """
    Exact all forms of numbers from a string with regex.
    https://github.com/MMMU-Benchmark/MMMU/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/eval_utils.py#L100
    """
    # Pattern for numbers with commas
    pattern_commas = r"-?\b\d{1,3}(?:,\d{3})+\b"
    # Pattern for scientific notation
    pattern_scientific = r"-?\d+(?:\.\d+)?[eE][+-]?\d+"
    # Pattern for simple numbers without commas
    pattern_simple = r"-?(?:\d+\.\d+|\.\d+|\d+\b)(?![eE][+-]?\d+)(?![,\d])"

    # Extract numbers with commas
    numbers_with_commas = re.findall(pattern_commas, string)
    # Extract numbers in scientific notation
    numbers_scientific = re.findall(pattern_scientific, string)
    # Extract simple numbers without commas
    numbers_simple = re.findall(pattern_simple, string)

    # Combine all extracted numbersz
    all_numbers = numbers_with_commas + numbers_scientific + numbers_simple
    return all_numbers


def check_is_number(string):
    """
    Check if the given string a number.
    https://github.com/MMMU-Benchmark/MMMU/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/eval_utils.py#L65
    """
    try:
        float(string.replace(",", ""))
        return True
    except ValueError:
        # check if there's comma inside
        return False


def normalize_str(string):
    """
    Normalize the str to lower case and make them float numbers if possible.
    https://github.com/MMMU-Benchmark/MMMU/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/eval_utils.py#L76
    """
    # check if characters in the string

    # if number, numerize it.
    string = string.strip()

    is_number = check_is_number(string)

    if is_number:
        string = string.replace(",", "")
        string = float(string)
        # leave 2 decimal
        string = round(string, 2)
        return [string]
    else:  # it's likely to be a string
        # lower it
        string = string.lower()
        if len(string) == 1:
            return [" " + string, string + " "]  # avoid trivial matches
        return [string]


def parse_open_response_mmmu(response):
    """
    Parse the prediction from the generated response.
    Return a list of predicted strings or numbers.
    https://github.com/MMMU-Benchmark/MMMU/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/eval_utils.py#L122
    """

    # content = content.strip("\n").strip(".").strip(" ")
    def get_key_subresponses(response):
        key_responses = []
        response = response.strip().strip(".").lower()
        sub_responses = re.split(r"\.\s(?=[A-Z])|\n", response)
        indicators_of_keys = [
            "could be ",
            "so ",
            "is ",
            "thus ",
            "therefore ",
            "final ",
            "answer ",
            "result ",
        ]
        key_responses = []
        for index, resp in enumerate(sub_responses):
            # if last one, accept it's an equation (the entire response can be just one sentence with equation)
            if index == len(sub_responses) - 1:
                indicators_of_keys.extend(["="])
            shortest_key_response = None  # the shortest response that may contain the answer (tail part of the response)
            for indicator in indicators_of_keys:
                if indicator in resp:
                    if not shortest_key_response:
                        shortest_key_response = resp.split(indicator)[-1].strip()
                    else:
                        if len(resp.split(indicator)[-1].strip()) < len(shortest_key_response):
                            shortest_key_response = resp.split(indicator)[-1].strip()
                    # key_responses.append(resp.split(indicator)[1].strip())

            if shortest_key_response:
                # and it's not trivial
                if shortest_key_response.strip() not in [
                    ":",
                    ",",
                    ".",
                    "!",
                    "?",
                    ";",
                    ":",
                    "'",
                ]:
                    key_responses.append(shortest_key_response)
        if len(key_responses) == 0:  # did not found any
            return [response]
        return key_responses

    # pdb.set_trace()
    key_responses = get_key_subresponses(response)

    pred_list = key_responses.copy()  # keep the original string response
    for resp in key_responses:
        pred_list.extend(extract_numbers(resp))

    tmp_pred_list = []
    for i in range(len(pred_list)):
        tmp_pred_list.extend(normalize_str(pred_list[i]))
    pred_list = tmp_pred_list

    # remove duplicates
    pred_list = list(set(pred_list))

    return pred_list


def get_multi_choice_info_mmmu(options):
    """
    Given the list of options for multiple choice question
    Return the index2ans and all_choices
    https://github.com/MMMU-Benchmark/MMMU/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/data_utils.py#L54
    """

    start_chr = "A"
    all_choices = []
    index2ans = {}
    for i, option in enumerate(options):
        index2ans[chr(ord(start_chr) + i)] = option
        all_choices.append(chr(ord(start_chr) + i))

    return index2ans, all_choices


with open(Path(__file__).parent / "mmbench_en.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)

    mmbench_config = yaml.safe_load("".join(safe_data))

GPT_EVAL_MODEL_NAME = _get_gpt_eval_model(mmbench_config)
mmbench_evaluator = MMBench_Evaluator(sys_prompt=mmbench_config["metadata"]["sys_prompt"], API_KEY=API_KEY, API_URL=API_URL, model_version=GPT_EVAL_MODEL_NAME)


def mmbench_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def mmbench_cn_cc_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    option_candidate = ["A", "B", "C", "D", "E"]
    options_prompt, options_dict = mmbench_evaluator.create_options_prompt(doc, option_candidate)

    data = {
        # "img": doc["image"],
        "question": doc["question"],
        "answer": doc.get("answer", None),
        "options": options_prompt,
        "category": doc["category"],
        "options_dict": options_dict,
        "index": doc["index"],
        "source": doc["source"],
    }

    query_prompt = f"{data['question']} {data['options']}"

    if lmms_eval_specific_kwargs:
        query_prompt = f"{query_prompt}\n{lmms_eval_specific_kwargs['post_prompt']}"

    return query_prompt


def mmbench_cn_cc_process_results(doc, results):
    model_response = results[0].strip()
    data = {
        "gpt_eval_score": {
            "index": doc["index"],
            "question": doc.get("question", ""),
            "prediction": model_response,
            "answer": doc.get("answer", None),
            "source": doc["source"],
            "category": doc["category"],
        },
        "submission": {
            "index": doc["index"],
            "prediction": model_response,
            "answer": doc.get("answer", None),
            "source": doc["source"],
            "category": doc["category"],
        },
    }
    option_candidate = ["A", "B", "C", "D", "E"]
    for c in option_candidate:
        data["submission"][c] = doc.get(c, "nan")
        data["gpt_eval_score"][c] = doc.get(c, "nan")
    return data


def mmbench_cn_cc_aggregate_dev_results_eval(results, args):
    print(f"============= MMBench-CN(CC) Detailed Results =============")
    overall_acc, category_acc, l2_category_acc = mmbench_evaluator.eval_result(results, eval_method="openai")
    args.output_path = args.output_path if args.output_path else "./"
    file = generate_submission_file("mmbench_cn_cc_results.json", args)
    details_info = {
        "overall_acc": overall_acc,
        "category_acc": category_acc,
        "l2_category_acc": l2_category_acc,
    }
    with open(file, "w") as f:
        json.dump(details_info, f)
    return overall_acc * 100


def mmbench_cn_cc_aggregate_results(results, args):
    df = pd.DataFrame(results)
    args.output_path = args.output_path if args.output_path else "./"
    file = generate_submission_file("mmbench_cn_cc_results.xlsx", args)
    with pd.ExcelWriter(file) as writer:
        df.to_excel(writer, index=False)
    eval_logger.info(f"Saved results to {file}")



def mmbench_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    option_candidate = ["A", "B", "C", "D", "E"]
    options_prompt, options_dict = mmbench_evaluator.create_options_prompt(doc, option_candidate)

    data = {
        # "img": doc["image"],
        "question": doc["question"],
        "answer": doc.get("answer", None),
        "options": options_prompt,
        "category": doc["category"],
        "L2-category": doc["L2-category"],
        "options_dict": options_dict,
        "index": doc["index"],
        "hint": doc["hint"],
        "source": doc["source"],
        "split": doc["split"],
    }

    query_prompt = f"{data['hint']} {data['question']} {data['options']}" if pd.notna(data["hint"]) else f"{data['question']} {data['options']}"

    if lmms_eval_specific_kwargs:
        query_prompt = f"{query_prompt}\n{lmms_eval_specific_kwargs['post_prompt']}"

    return query_prompt


def mmbench_process_results(doc, results):
    model_response = results[0].strip()
    data = {
        "gpt_eval_score": {
            "index": doc["index"],
            "question": doc.get("question", ""),
            "prediction": model_response,
            "answer": doc.get("answer", None),
            "source": doc["source"],
            "split": doc["split"],
            "category": doc["category"],
            "L2-category": doc["L2-category"],
        },
        "submission": {
            "index": doc["index"],
            "prediction": model_response,
            "source": doc["source"],
            "split": doc["split"],
            "category": doc["category"],
            "L2-category": doc["L2-category"],
        },
    }
    option_candidate = ["A", "B", "C", "D", "E"]
    for c in option_candidate:
        data["submission"][c] = doc.get(c, "nan")
        data["gpt_eval_score"][c] = doc.get(c, "nan")
    return data


def mmbench_aggregate_dev_results_eval_cn(results, args):
    print(f"============= MMBench-CN(Dev) Detailed Results =============")
    overall_acc, category_acc, l2_category_acc = mmbench_evaluator.eval_result(results, eval_method="openai")
    args.output_path = args.output_path if args.output_path else "./"
    file = generate_submission_file("mmbench_cn_dev_results.json", args)
    details_info = {
        "overall_acc": overall_acc,
        "category_acc": category_acc,
        "l2_category_acc": l2_category_acc,
    }
    with open(file, "w") as f:
        json.dump(details_info, f)
    return overall_acc * 100


def mmbench_aggregate_dev_results(results, args):
    df = pd.DataFrame(results)
    args.output_path = args.output_path if args.output_path else "./"
    excel_write_path = generate_submission_file("mmbench_cn_dev_results.xlsx", args)
    with pd.ExcelWriter(excel_write_path) as writer:
        df.to_excel(writer, index=False)
    eval_logger.info(f"Saved results to {excel_write_path}")


def mmbench_aggregate_test_results_cn(results, args):
    df = pd.DataFrame(results)
    args.output_path = args.output_path if args.output_path else "./"
    excel_write_path = generate_submission_file("mmbench_cn_test_results.xlsx", args)
    with pd.ExcelWriter(excel_write_path) as writer:
        df.to_excel(writer, index=False)
    eval_logger.info(f"Saved results to {excel_write_path}")


def mmbench_aggregate_dev_results_eval_en(results, args):
    print(f"============= MMBench-EN(Dev) Detailed Results =============")
    overall_acc, category_acc, l2_category_acc = mmbench_evaluator.eval_result(results, eval_method="openai")
    args.output_path = args.output_path if args.output_path else "./"
    file = generate_submission_file("mmbench_en_dev_results.json", args)
    details_info = {
        "overall_acc": overall_acc,
        "category_acc": category_acc,
        "l2_category_acc": l2_category_acc,
    }
    with open(file, "w") as f:
        json.dump(details_info, f)
    return overall_acc * 100

def mmbench_aggregate_dev_results_submission(results, args):
    df = pd.DataFrame(results)
    args.output_path = args.output_path if args.output_path else "./"
    excel_write_path = generate_submission_file("mmbench_en_dev_results.xlsx", args)
    with pd.ExcelWriter(excel_write_path) as writer:
        df.to_excel(writer, index=False)
    eval_logger.info(f"Saved results to {excel_write_path}")

def mmbench_aggregate_test_results_en(results, args):
    df = pd.DataFrame(results)
    args.output_path = args.output_path if args.output_path else "./"
    excel_write_path = generate_submission_file("mmbench_en_test_results.xlsx", args)
    with pd.ExcelWriter(excel_write_path) as writer:
        df.to_excel(writer, index=False)
    eval_logger.info(f"Saved results to {excel_write_path}")


OPTIONS = ["A", "B", "C", "D", "E"]
with open(Path(__file__).parent / "nextqa_oe_test.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)

    nextqa_config = yaml.safe_load("".join(safe_data))
if nextqa_config["metadata"]["load_package"]:
    try:
        from pywsd.utils import lemmatize_sentence
    except ImportError:
        eval_logger.debug("pywsd not installed. Please install pywsd to use this module. You can install it by running 'pip install pywsd'")

    try:
        import nltk
        from nltk.corpus import wordnet
        from nltk.tokenize import word_tokenize

        nltk.download("averaged_perceptron_tagger", quiet=True)
        nltk.download("wordnet", quiet=True)
        nltk.download("punkt", quiet=True)
    except ImportError:
        eval_logger.debug("nltk not installed. Please install nltk to use this module. You can install it by running 'pip install nltk'")

stopwords = set(pd.read_csv(Path(__file__).parent / "stopwords.csv").squeeze())

nextqa_cache_dir = os.path.join(nextqa_config["dataset_path"], "NExTVideo")


def nextqa_doc_to_visual(doc):
    return [get_video(nextqa_cache_dir, doc["video"])]


def nextqa_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["question"].strip()
    if "pre_prompt" in lmms_eval_specific_kwargs and lmms_eval_specific_kwargs["pre_prompt"] != "":
        question = f"{lmms_eval_specific_kwargs['pre_prompt']}{question}"
    if "post_prompt" in lmms_eval_specific_kwargs and lmms_eval_specific_kwargs["post_prompt"] != "":
        question = f"{question}{lmms_eval_specific_kwargs['post_prompt']}"
    return question


def nextqa_doc_to_text_mc(doc, lmms_eval_specific_kwargs=None):
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    question = [doc["question"].strip()]
    for i in range(5):
        question.append(f"{OPTIONS[i]}. {doc[f'a{i}'].strip()}")
    question = "\n".join(question)
    if "pre_prompt" in lmms_eval_specific_kwargs and lmms_eval_specific_kwargs["pre_prompt"] != "":
        question = f"{lmms_eval_specific_kwargs['pre_prompt']}{question}"
    if "post_prompt" in lmms_eval_specific_kwargs and lmms_eval_specific_kwargs["post_prompt"] != "":
        question = f"{question}{lmms_eval_specific_kwargs['post_prompt']}"
    return question


def nextqa_mc_process_results(doc, results):
    pred = results[0]
    index2ans, all_choices = get_multi_choice_info_nextqa(doc)
    parsed_pred = parse_multi_choice_response_nextqa(pred, all_choices, index2ans)
    return {
        "exact_match": parsed_pred == OPTIONS[doc["answer"]],
    }


def parse_multi_choice_response_nextqa(response, all_choices, index2ans):
    """
    Parse the prediction from the generated response.
    Return the predicted index e.g., A, B, C, D.
    https://github.com/MMMU-Benchmark/MMMU/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/eval_utils.py#L10
    """
    for char in [",", ".", "!", "?", ";", ":", "'"]:
        response = response.strip(char)
    response = " " + response + " "  # add space to avoid partial match

    index_ans = True
    ans_with_brack = False
    candidates = []
    for choice in all_choices:  # e.g., (A) (B) (C) (D)
        if f"({choice})" in response:
            candidates.append(choice)
            ans_with_brack = True

    if len(candidates) == 0:
        for choice in all_choices:  # e.g., A B C D
            if f"{choice} " in response:
                candidates.append(choice)

    if len(candidates) == 0:
        for choice in all_choices:  # e.g., A. B. C. D.
            if f"{choice}." in response:
                candidates.append(choice)

    # if all above doesn't get candidates, check if the content is larger than 5 tokens and try to parse the example
    if len(candidates) == 0 and len(response.split()) > 5:
        for index, ans in index2ans.items():
            if ans.lower() in response.lower():
                candidates.append(index)
                index_ans = False  # it's content ans.

    if len(candidates) == 0:  # still not get answer, randomly choose one.
        pred_index = random.choice(all_choices)
    elif len(candidates) > 1:
        start_indexes = []
        if index_ans:
            if ans_with_brack:
                for can in candidates:
                    index = response.rfind(f"({can})")
                    start_indexes.append(index)  # -1 will be ignored anyway
                # start_indexes = [generated_response.index(f'({can})') for can in candidates]
            else:
                for can in candidates:
                    index = response.rfind(f" {can} ")
                    start_indexes.append(index)
        else:
            for can in candidates:
                index = response.lower().rfind(index2ans[can].lower())
                start_indexes.append(index)
        # get the last one
        pred_index = candidates[np.argmax(start_indexes)]
    else:  # if only one candidate, use it.
        pred_index = candidates[0]

    return pred_index


def nextqa_doc_to_target(doc):
    return doc["answer"]


def remove_stop(sentence):
    sentence.replace("</s>", "")  # video-llava
    words = lemmatize_sentence(sentence)
    words = [w for w in words if not w in stopwords]
    return " ".join(words)


def get_multi_choice_info_nextqa(doc):
    all_choices = []
    index2ans = {}
    for i in range(5):
        index2ans[OPTIONS[i]] = doc[f"a{i}"].strip()
        all_choices.append(OPTIONS[i])

    return index2ans, all_choices


####################### WUPS ################################
# The following code copied from                            #
# https://github.com/doc-doc/NExT-OE/blob/main/metrics.py   #
#############################################################

# ====================================================
# @Time    : 13/9/20 4:19 PM
# @Author  : Xiao Junbin
# @Email   : junbin@comp.nus.edu.sg
# @File    : metrics.py
# ====================================================


def wup(word1, word2, alpha):
    """
    calculate the wup similarity
    :param word1:
    :param word2:
    :param alpha:
    :return:
    """
    # print(word1, word2)
    if word1 == word2:
        return 1.0

    w1 = wordnet.synsets(word1)
    w1_len = len(w1)
    if w1_len == 0:
        return 0.0
    w2 = wordnet.synsets(word2)
    w2_len = len(w2)
    if w2_len == 0:
        return 0.0

    # match the first
    word_sim = w1[0].wup_similarity(w2[0])
    if word_sim is None:
        word_sim = 0.0

    if word_sim < alpha:
        word_sim = 0.1 * word_sim
    return word_sim


def wups(words1, words2, alpha):
    """

    :param pred:
    :param truth:
    :param alpha:
    :return:
    """
    sim = 1.0
    flag = False
    for w1 in words1:
        max_sim = 0
        for w2 in words2:
            word_sim = wup(w1, w2, alpha)
            if word_sim > max_sim:
                max_sim = word_sim
        if max_sim == 0:
            continue
        sim *= max_sim
        flag = True
    if not flag:
        sim = 0.0
    return sim


def get_wups(pred, truth, alpha):
    """
    calculate the wups score
    :param pred:
    :param truth:
    :return:
    """
    pred = word_tokenize(pred)
    truth = word_tokenize(truth)
    item1 = wups(pred, truth, alpha)
    item2 = wups(truth, pred, alpha)
    value = min(item1, item2)
    return value

def nextqa_process_results(doc, results):
    pred = results[0]
    answer = doc["answer"]
    pred_ans = remove_stop(pred)
    gt_ans = remove_stop(answer)
    qtype = doc["type"]
    if qtype == "TP":
        qtype = "TN"
    add_ref_ans = doc["additional_ref_answer"]
    if add_ref_ans:
        add_ref_ans = remove_stop(add_ref_ans)
        if qtype == "DC" or qtype == "DB":
            cur_0 = 1 if pred_ans == gt_ans or pred_ans == add_ref_ans else 0
            cur_9 = cur_0
        else:
            cur_0 = max(get_wups(pred_ans, gt_ans, 0), get_wups(pred_ans, add_ref_ans, 0))
            cur_9 = max(get_wups(pred_ans, gt_ans, 0.9), get_wups(pred_ans, add_ref_ans, 0))
    else:
        if qtype == "DC" or qtype == "DB":
            cur_0 = 1 if pred_ans == gt_ans else 0
            cur_9 = cur_0
        else:
            cur_0 = get_wups(pred_ans, gt_ans, 0)
            cur_9 = get_wups(pred_ans, gt_ans, 0.9)
    return {"WUPS": {"0": cur_0, "0.9": cur_9, "qtype": qtype}}


def nextqa_aggregate_results(results):
    qtypes = ["CW", "CH", "TN", "TC", "DB", "DC", "DL", "DO"]
    num = {"CW": 0, "CH": 0, "TN": 0, "TC": 0, "DB": 0, "DC": 0, "DL": 0, "DO": 0}
    over_num = {"C": 0, "T": 0, "D": 0}
    wups0 = {"CW": 0, "CH": 0, "TN": 0, "TC": 0, "DB": 0, "DC": 0, "DL": 0, "DO": 0}
    wups9 = {"CW": 0, "CH": 0, "TN": 0, "TC": 0, "DB": 0, "DC": 0, "DL": 0, "DO": 0}
    ref_num = 0
    for result in results:
        qtype = result["qtype"]
        num[qtype] += 1
        over_num[qtype[0]] += 1
        ref_num += 1
        cur_0 = result["0"]
        cur_9 = result["0.9"]
        wups0[qtype] += cur_0
        wups9[qtype] += cur_9

    wups0_all = wups9_all = 0
    wups0_e = wups0_t = wups0_c = 0
    for qtype in qtypes:
        wups0_all += wups0[qtype]
        wups9_all += wups9[qtype]
        if qtype[0] == "C":
            wups0_e += wups0[qtype]
        if qtype[0] == "T":
            wups0_t += wups0[qtype]
        if qtype[0] == "D":
            wups0_c += wups0[qtype]

        if num[qtype] != 0:
            wups0[qtype] = wups0[qtype] / num[qtype]
            wups9[qtype] = wups9[qtype] / num[qtype]
        else:
            wups0[qtype] = 0
            wups9[qtype] = 0

    wups0_all /= ref_num
    wups9_all /= ref_num

    for k in qtypes:
        wups0[k] = wups0[k] * 100
        wups9[k] = wups9[k] * 100

    wups0_all *= 100

    return wups0_all

NUM_SECONDS_TO_SLEEP = 5 
with open(Path(__file__).parent / "activitynetqa_generation.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        if "!function" not in line:
            safe_data.append(line)

    activitynetqa_config = yaml.safe_load("".join(safe_data))
HF_HOME = os.environ["HF_HOME"]
activitynetqa_cache_dir = activitynetqa_config["dataset_path"]
activitynetqa_cache_dir = os.path.join(activitynetqa_cache_dir, "all_test")

# Pass in video path here
# Can only work correctly with video dataset
def activitynetqa_doc_to_visual(doc):
    video_path = os.path.join(activitynetqa_cache_dir, f"v_{doc['video_name']}.mp4")
    extensions = ["mp4", "webm", "mkv"]
    for ext in extensions:
        modified_path = video_path.replace("mp4", ext)
        if os.path.exists(modified_path):
            return [modified_path]
    sys.exit(f"video path:{video_path} does not exist, please check")

# This is the place where format the question
def activitynetqa_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    pre_prompt = ""
    post_prompt = ""
    if "pre_prompt" in lmms_eval_specific_kwargs:
        pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    if "post_prompt" in lmms_eval_specific_kwargs:
        post_prompt = lmms_eval_specific_kwargs["post_prompt"]

    raw_question = doc["question"]
    question = raw_question.capitalize() + "?"

    # type_specific_prompts = {
    #     '3': "Please answer with 'yes' or 'no'.",
    #     '4': "Please state the color as a single word.",
    #     '7': "Please give the numerical answer."
    # }

    # doc_type = str(doc['type'])
    # type_specific_prompt = type_specific_prompts.get(doc_type, "")

    # return f"{pre_prompt}{question} {type_specific_prompt}{post_prompt}"
    return f"{pre_prompt}{question}{post_prompt}"

def activitynetqa_doc_to_answer(doc):
    return doc["answer"]

def get_eval(question, answer, pred, max_tokens: int, retries: int = 5):
    messages = [
        {
            "role": "system",
            "content": (
                "You are an intelligent chatbot designed for evaluating the correctness of generative outputs for question-answer pairs. "
                "Your task is to compare the predicted answer with the correct answer and determine if they match meaningfully. Here's how you can accomplish the task:"
                "------"
                "##INSTRUCTIONS: "
                "- Focus on the meaningful match between the predicted answer and the correct answer.\n"
                "- Consider synonyms or paraphrases as valid matches.\n"
                "- Evaluate the correctness of the prediction compared to the answer."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Please evaluate the following video-based question-answer pair:\n\n"
                f"Question: {question}\n"
                f"Correct Answer: {answer}\n"
                f"Predicted Answer: {pred}\n\n"
                "Provide your evaluation only as a yes/no and score where the score is an integer value between 0 and 5, with 5 indicating the highest meaningful match. "
                "Please generate the response in the form of a Python dictionary string with keys 'pred' and 'score', where value of 'pred' is  a string of 'yes' or 'no' and value of 'score' is in INTEGER, not STRING."
                "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
                "For example, your response should look like this: {'pred': 'yes', 'score': 4.8}."
            ),
        },
    ]

    # Keep behavior aligned with the original: fixed temperature=0, respect max_tokens
    create_kwargs = {
        "model": _get_gpt_eval_model(activitynetqa_config),
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
    }

    last_err = None
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(**create_kwargs)
            # Keep original behavior: return non-empty content ASAP
            content = _chat_completion_content(response)
            if content != "":
                # Keep original second return as "response_data['model']"
                # SDK response may expose .model; otherwise fall back.
                used_model = getattr(response, "model", _get_gpt_eval_model(activitynetqa_config))
                return content, used_model

        except Exception as e:
            last_err = e
            # Preserve original logging style as closely as possible
            try:
                eval_logger.error(f"Unexpected error on attempt {attempt + 1}: {e}")
            except Exception:
                pass

        # Sleep between retries like original
        if attempt < retries - 1:
            time.sleep(NUM_SECONDS_TO_SLEEP)
        else:
            try:
                eval_logger.error(f"All {retries} attempts failed. Last error message: {last_err}")
            except Exception:
                pass
            return "", ""

    return "", ""

def parse_score(review):
    try:
        # Convert the string representation of a dictionary to an actual dictionary
        review = "{" + review.split("{")[1].split("}")[0] + "}"
        review_dict = ast.literal_eval(review)
        # import pdb;pdb.set_trace()
        score_match = review_dict["score"]
        score = int(score_match)
        pred = review_dict["pred"]
        if "yes" in pred.lower():
            pred = "yes"
        elif "no" in pred.lower():
            pred = "no"
        # pred = review_dict.get("pred", "no")
        # score = review_dict.get("score", 0)
        return [pred, score]
    except SyntaxError as e:
        eval_logger.error(f"Syntax error parsing the review string: {e}. Review content: {review}")
    except ValueError as e:
        eval_logger.error(f"Value error parsing the review string: {e}. Review content: {review}")
    except Exception as e:
        eval_logger.error(f"Unexpected error parsing the review string: {e}. Review content: {review}")

def activitynetqa_process_results(doc, result):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary
    """
    try:
        question = doc["question"]
        answer = doc["answer"]
        pred = result[0]

        # Try to get GPT evaluation, but fall back to simple matching if API fails
        try:
            review, model_name = get_eval(question, answer, pred, 64)
            scores = parse_score(review)
        except Exception as e:
            eval_logger.warning(f"GPT evaluation failed, using simple matching: {e}")
            # Simple string matching fallback
            if pred.lower().strip() == answer.lower().strip():
                scores = ["yes", 5]
            else:
                scores = ["no", 0]
    except Exception as e:
        eval_logger.error(f"Error for Question ID: {doc.get('question_id', 'Unknown')}: {e}")
        review = "Failed to Get a Proper Review."
        model_name = "Failed Request"
        scores = ["no", 0]

    return {
        "gpt_eval_score": {"video_name": doc["video_name"], "question_id": doc["question_id"], "type": doc["type"], "Correctness": scores[0], "score": scores[1]},
        "gpt_eval_accuracy": {"video_name": doc["video_name"], "question_id": doc["question_id"], "type": doc["type"], "Correctness": scores[0], "score": scores[1]},
    }

def activitynetqa_gpt_eval(results, args):
    """
    Process the result file containing predictions, score them using GPT,
    and save the results with added scores and correctness fields to a new file.

    Args:
        result_file_path: path to the JSON file with results to be evaluated
        eval_file_path: path to save the JSON file with evaluated results
    """

    evaluated_results = []

    # Process each result to generate scores
    for data_dict in results:
        try:
            question = data_dict.get("Q", "")
            answer = data_dict.get("A", "")
            pred = data_dict.get("pred", "")

            # Assume get_eval returns a review and the model name, and parse_score parses this review
            review, model_name = get_eval(question, answer, pred, 64)
            scores = parse_score(review)
        except Exception as e:
            eval_logger.error(f"Error for Question ID: {data_dict.get('question_id', 'Unknown')}: {e}")
            review = "Failed to Get a Proper Review."
            model_name = "Failed Request"
            scores = ["no", 0]

        # Update the dictionary with the new entries
        updated_dict = {"video_name": data_dict["video_name"], "Correctness": scores[0], "score": scores[1], "Q": question, "A": answer, "pred": pred, "question_id": data_dict.get("question_id"), "type": data_dict.get("type")}
        evaluated_results.append(updated_dict)

    return evaluated_results

# Factory into different aggregate
def activitynetqa_aggregate_score(results, args):
    yes_count = 0
    no_count = 0
    total_score = 0

    # Iterate over the results to count correctness and sum scores
    for result_dict in results:
        if "yes" in result_dict["Correctness"].lower():
            yes_count += 1
        elif "no" in result_dict["Correctness"].lower():
            no_count += 1

        total_score += int(result_dict["score"])

    # Calculate accuracy and average score
    accuracy = yes_count / (yes_count + no_count) if (yes_count + no_count) > 0 else 0
    average_score = total_score / len(results) if results else 0
    eval_logger.info(f"Accuracy: {accuracy}")
    eval_logger.info(f"Average Score: {average_score}")
    return average_score

def activitynetqa_aggregate_accuracy(results, args):
    yes_count = 0
    no_count = 0
    total_score = 0

    # Iterate over the results to count correctness and sum scores
    for result_dict in results:
        if "yes" in result_dict["Correctness"].lower():
            yes_count += 1
        elif "no" in result_dict["Correctness"].lower():
            no_count += 1

        total_score += int(result_dict["score"])

    # Calculate accuracy and average score
    accuracy = yes_count / (yes_count + no_count) if (yes_count + no_count) > 0 else 0
    average_score = total_score / len(results) if results else 0
    eval_logger.info(f"Accuracy: {accuracy}")
    eval_logger.info(f"Average Score: {average_score}")
    return accuracy * 100


def timestamp_to_seconds(timestamp):
    # Split the timestamp into hours, minutes, and seconds
    h, m, s = timestamp.split(":")
    # Convert hours, minutes, and total seconds (including fractions) to float and compute total seconds
    total_seconds = int(h) * 3600 + int(m) * 60 + float(s)
    return total_seconds


def load_video(video_file, duration, max_num_frames=16):
    from decord import VideoReader

    vr = VideoReader(video_file, ctx=cpu(0), num_threads=1)
    fps = vr.get_avg_fps()
    total_valid_frames = int(duration * fps)
    num_frames = min(max_num_frames, int(duration))

    frame_indices = [int(total_valid_frames / num_frames) * i for i in range(num_frames)]

    frames = vr.get_batch(frame_indices)
    if isinstance(frames, torch.Tensor):
        frames = frames.numpy()
    else:
        frames = frames.asnumpy()
    frame_timestamps = [frame_index / fps for frame_index in frame_indices]

    return [Image.fromarray(fr).convert("RGB") for fr in frames]


def compute_frame_timestamps(duration, max_num_frames=16):
    if duration > max_num_frames:
        return [duration / max_num_frames * i for i in range(max_num_frames)]
    else:
        return [i for i in range(int(duration))]


def insert_subtitles_into_frames(frame_timestamps, subtitles, starting_timestamp_for_subtitles, duration):
    interleaved_list = []
    cur_i = 0

    for subtitle in subtitles:
        if "timestamp" in subtitle:
            start, end = subtitle["timestamp"]

            if not isinstance(end, float):
                end = duration

            start -= starting_timestamp_for_subtitles
            end -= starting_timestamp_for_subtitles

            subtitle_timestamp = (start + end) / 2
            subtitle_text = subtitle["text"]
        else:
            start, end = subtitle["start"], subtitle["end"]
            start = timestamp_to_seconds(start)
            end = timestamp_to_seconds(end)
            start -= starting_timestamp_for_subtitles
            end -= starting_timestamp_for_subtitles

            subtitle_timestamp = (start + end) / 2
            subtitle_text = subtitle["line"]

        for i, frame_timestamp in enumerate(frame_timestamps[cur_i:]):
            if frame_timestamp <= subtitle_timestamp:
                # print("frame:", frame_timestamp)
                interleaved_list.append("<image>")
                cur_i += 1
            else:
                break

        if end - start < 1:
            end = subtitle_timestamp + 0.5
            start = subtitle_timestamp - 0.5

        covering_frames = False
        for frame_timestamp in frame_timestamps:
            if frame_timestamp < end and frame_timestamp > start:
                covering_frames = True
                break

        if covering_frames:
            # print("subtitle:", subtitle_timestamp, start, end)
            interleaved_list.append(subtitle_text)
        else:
            pass
            # print("leaving out subtitle:", start, end)

    for i, frame_timestamp in enumerate(frame_timestamps[cur_i:]):
        # print(frame_timestamp)
        interleaved_list.append("<image>")

    return "\n".join(interleaved_list)


def longvideobench_doc_to_text(doc, lmms_eval_specific_kwargs):
    candidates = []

    for i in range(5):
        candidate = doc.get(f"option{i}")
        if candidate != "N/A":
            candidates.append(candidate)

    question = doc["question"] + "\n" + "\n".join([". ".join([chr(ord("A") + i), candidate]) for i, candidate in enumerate(candidates)])
    pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    post_prompt = lmms_eval_specific_kwargs["post_prompt"]

    if lmms_eval_specific_kwargs.get("insert_interleave_subtitles", False):
        with open(Path(__file__).parent / "longvideobench_test.yaml", "r") as f:
            raw_data = f.readlines()
            safe_data = []
            for i, line in enumerate(raw_data):
                # remove function definition since yaml load cannot handle it
                if "!function" not in line:
                    safe_data.append(line)
            safe_data = yaml.safe_load("".join(safe_data))
        base_cache_dir = safe_data["dataset_path"]
        subtitle_subdir_name = safe_data["dataset_kwargs"].get("subtitle_subdir", "subtitles")
        cache_dir = os.path.join(base_cache_dir, subtitle_subdir_name)
        with open(os.path.join(cache_dir, doc["subtitle_path"])) as f:
            subtitles = json.load(f)

        max_num_frames = safe_data["dataset_kwargs"].get("max_num_frames", 16)

        frame_timestamps = compute_frame_timestamps(doc["duration"], max_num_frames)
        interleaved_prefix = insert_subtitles_into_frames(frame_timestamps, subtitles, doc["starting_timestamp_for_subtitles"], doc["duration"])
        return f"{pre_prompt}{interleaved_prefix}\n{question}\n{post_prompt}"
    else:
        return f"{pre_prompt}{question}\n{post_prompt}"


def longvideobench_doc_to_visual_v(doc):
    with open(Path(__file__).parent / "longvideobench_test.yaml", "r") as f:
        raw_data = f.readlines()
        safe_data = []
        for i, line in enumerate(raw_data):
            # remove function definition since yaml load cannot handle it
            if "!function" not in line:
                safe_data.append(line)
        safe_data = yaml.safe_load("".join(safe_data))
    base_cache_dir = safe_data["dataset_path"]
    vid_subdir_name = safe_data["dataset_kwargs"].get("video_subdir", "videos/")
    cache_dir = os.path.join(base_cache_dir, vid_subdir_name)
    video_path = doc["video_path"]
    video_path = os.path.join(cache_dir, video_path)
    return [video_path]


def longvideobench_doc_to_visual_i(doc):
    with open(Path(__file__).parent / "longvideobench_test.yaml", "r") as f:
        raw_data = f.readlines()
        safe_data = []
        for i, line in enumerate(raw_data):
            # remove function definition since yaml load cannot handle it
            if "!function" not in line:
                safe_data.append(line)
    base_cache_dir = yaml.safe_load("".join(safe_data))["dataset_path"]
    vid_subdir_name = yaml.safe_load("".join(safe_data))["dataset_kwargs"].get("video_subdir", "videos/")
    cache_dir = os.path.join(base_cache_dir, vid_subdir_name)
    video_path = doc["video_path"]
    video_path = os.path.join(cache_dir, video_path)
    max_num_frames = yaml.safe_load("".join(safe_data))["dataset_kwargs"].get("max_num_frames", 16)
    return load_video(video_path, doc["duration"], max_num_frames)


def parse_multi_choice_response_longvideobench(response, all_choices, index2ans):
    """
    Changed from MMMU-style complex parsing into simple parsing.
    Fixed to avoid 'D. A book' be parsed as A.
    Same as original LongVideoBench paper (from author Haoning Wu), if parsing failed, it will assign a random choice to model.
    """
    s = response.strip()
    answer_prefixes = [
        "The best answer is",
        "The correct answer is",
        "The answer is",
        "The answer",
        "The best option is",
        "The correct option is",
        "Best answer:",
        "Best option:",
    ]
    for answer_prefix in answer_prefixes:
        s = s.replace(answer_prefix, "")

    if len(s.split()) > 10 and not re.search("[ABCDE]", s):
        return random.choice(all_choices)

    matches = re.search(r"[ABCDE]", s)
    if matches is None:
        return random.choice(all_choices)
    return matches[0]


def evaluate_longvideobench(samples):
    pred_correct = 0
    judge_dict = dict()
    for sample in samples:
        gold_i = sample["answer"]
        pred_i = sample["parsed_pred"]
        correct = eval_multi_choice(gold_i, pred_i)

        if correct:
            judge_dict[sample["id"]] = "Correct"
            pred_correct += 1
        else:
            judge_dict[sample["id"]] = "Wrong"

    if len(samples) == 0:
        return {"acc": 0}
    return judge_dict, {"acc": pred_correct / len(samples)}


def longvideobench_process_results(doc, results):
    pred = results[0]
    all_choices = []
    index2ans = {}
    for i in range(5):
        option = doc.get(f"option{i}")
        if option == "N/A":
            break
        index2ans[chr(ord("A") + i)] = option
        all_choices.append(chr(ord("A") + i))

    parsed_pred = parse_multi_choice_response_longvideobench(pred, all_choices, index2ans)
    id = doc["id"]
    lvb_acc = {"id": id, "duration_group": doc["duration_group"], "question_category": doc["question_category"], "answer": chr(ord("A") + doc["correct_choice"]), "parsed_pred": parsed_pred}
    return {
        "lvb_acc": lvb_acc,
        "submission": {
            id: pred,
        },
    }

def longvideobench_aggregate_results(results, args):
    evaluation_result = {}
    subset_to_eval_samples = defaultdict(list)
    for result in results:
        subset_to_eval_samples[result["duration_group"]].append(result)
        subset_to_eval_samples[result["question_category"]].append(result)
    for subset, sub_eval_samples in subset_to_eval_samples.items():
        judge_dict, metric_dict = evaluate_longvideobench(sub_eval_samples)
        metric_dict.update({"num_example": len(sub_eval_samples)})
        evaluation_result[subset] = metric_dict
    printable_results = {}

    for cat_name, cat_results in evaluation_result.items():
        printable_results[cat_name] = {
            "num": int(cat_results["num_example"]),
            "acc": round(cat_results["acc"], 5),
        }
    all_ins_acc = calculate_ins_level_acc(evaluation_result)
    printable_results["Overall"] = {
        "num": sum([cat_results["num_example"] for cat_results in evaluation_result.values()]),
        "acc": round(all_ins_acc, 5),
    }
    eval_logger.info(printable_results)
    return printable_results["Overall"]["acc"]


def longvideobench_aggregate_results_for_submission(results, args):
    args.output_path = args.output_path if args.output_path else "./"
    path = generate_submission_file("longvideobench_test_for_submission.json", args)
    results_dict = {list(item.keys())[0]: list(item.values())[0] for item in results}
    with open(path, "w") as f:
        json.dump(results_dict, f)
    eval_logger.info(f"Results saved to {path}.")

with open(Path(__file__).parent / "tempcompass_yes_no.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)

    tempcompass_yes_no_config = yaml.safe_load("".join(safe_data))

with open(Path(__file__).parent / "tempcompass_captioning.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        if "!function" not in line:
            safe_data.append(line)
    tempcompass_captioning_config = yaml.safe_load("".join(safe_data))

with open(Path(__file__).parent / "tempcompass_caption_matching.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        if "!function" not in line:
            safe_data.append(line)
    tempcompass_caption_matching_config = yaml.safe_load("".join(safe_data))

with open(Path(__file__).parent / "tempcompass_mc.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        if "!function" not in line:
            safe_data.append(line)
    tempcompass_mc_config = yaml.safe_load("".join(safe_data))

# We will unzip all the zip files
# To HF HOME cache dir
# And load it here
cache_dir_tempcompass = tempcompass_yes_no_config["dataset_path"]
cache_dir_tempcompass = os.path.join(cache_dir_tempcompass, "videos")

# Pass in video path here
# Can only work correctly with video llm
def tempcompass_doc_to_visual(doc):
    video_path = doc["video_id"] + ".mp4"
    video_path = os.path.join(cache_dir_tempcompass, video_path)
    if os.path.exists(video_path):
        video_path = video_path
    else:
        sys.exit(f"video path:{video_path} does not exist, please check")
    return [video_path]


# This is the place where you format your question
def tempcompass_doc_to_text_multi_choice(doc, lmms_eval_specific_kwargs=None):
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    pre_prompt = ""
    post_prompt = ""
    if "pre_prompt" in lmms_eval_specific_kwargs:
        pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    if "post_prompt" in lmms_eval_specific_kwargs:
        post_prompt = lmms_eval_specific_kwargs["post_prompt"][0]["multi-choice"]

    question = doc["question"]
    return f"{pre_prompt}{question}{post_prompt}"


def tempcompass_doc_to_text_yes_no(doc, lmms_eval_specific_kwargs=None):
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    pre_prompt = ""
    post_prompt = ""
    if "pre_prompt" in lmms_eval_specific_kwargs:
        pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    if "post_prompt" in lmms_eval_specific_kwargs:
        post_prompt = lmms_eval_specific_kwargs["post_prompt"][1]["yes_no"]

    question = doc["question"]
    return f"{pre_prompt}{question}{post_prompt}"


def tempcompass_doc_to_text_caption_matching(doc, lmms_eval_specific_kwargs=None):
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    pre_prompt = ""
    post_prompt = ""
    if "pre_prompt" in lmms_eval_specific_kwargs:
        pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    if "post_prompt" in lmms_eval_specific_kwargs:
        post_prompt = lmms_eval_specific_kwargs["post_prompt"][2]["caption_matching"]

    question = doc["question"]
    return f"{pre_prompt}{question}{post_prompt}"


def tempcompass_doc_to_text_captioning(doc, lmms_eval_specific_kwargs=None):
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    pre_prompt = ""
    post_prompt = ""
    if "pre_prompt" in lmms_eval_specific_kwargs:
        pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    if "post_prompt" in lmms_eval_specific_kwargs:
        post_prompt = lmms_eval_specific_kwargs["post_prompt"][3]["captioning"]

    question = doc["question"]
    return f"{pre_prompt}{question}{post_prompt}"


def tempcompass_doc_to_answer(doc):
    return doc["answer"]


# Process result for multi_choice
def tempcompass_process_results_multi_choice(doc, result):
    pred = result[0]
    rating = 0
    match_success = True
    chatgpt_response = None

    # Some hand-crafted matching rules
    if pred == doc["answer"]:
        rating = 1
    elif pred in ["A", "B", "C", "D"]:
        rating = 1 if pred == doc["answer"][0] else 0
    elif any(pred.startswith(prefix) for prefix in ["A.", "B.", "C.", "D."]):
        rating = 1 if pred.split(".")[0] == doc["answer"][0] else 0
    elif any(pred.startswith(prefix) for prefix in ["A)", "B)", "C)", "D)"]):
        rating = 1 if pred.split(")")[0] == doc["answer"][0] else 0
    else:
        # Fail to match answer in the video-llm response. Use ChatGPT to evaluate.
        match_success = False

        base_prompt = """
        You will receive a multi-choice question, the ground-truth answer and the prediction from a question answering (QA) model. \
        Your task is to determine whether QA model prediction is correct, based on the question and ground-truth answer. \
        If the prediction is correct, respond "Correct". If the prediction is incorrect, respond "Incorrect".
        """
        prompt = f"""{base_prompt}\nMulti-Choice Question:\n{doc["question"]}\nGround-Truth Answer: {doc["answer"]}\nModel Prediction: {pred}"""
        chatgpt_response, rating = get_eval_result(
            prompt,
            model=_get_gpt_eval_model(tempcompass_mc_config),
        )

    if chatgpt_response:
        return {
            "avg_accuracy": {
                "video_id": doc["video_id"],
                "question": doc["question"],
                "gt-answer": doc["answer"],
                "video-llm-prediction": pred,
                "match_success": match_success,
                "rating": rating,
                "chatgpt_response": chatgpt_response,
                "dim": doc["dim"],
            },
            doc["dim"]
            + "_accuracy": {
                "video_id": doc["video_id"],
                "question": doc["question"],
                "gt-answer": doc["answer"],
                "video-llm-prediction": pred,
                "match_success": match_success,
                "rating": rating,
                "chatgpt_response": chatgpt_response,
                "dim": doc["dim"],
            },
        }
    else:
        return {
            "avg_accuracy": {"video_id": doc["video_id"], "question": doc["question"], "gt-answer": doc["answer"], "video-llm-prediction": pred, "match_success": match_success, "rating": rating, "dim": doc["dim"]},
            doc["dim"] + "_accuracy": {"video_id": doc["video_id"], "question": doc["question"], "gt-answer": doc["answer"], "video-llm-prediction": pred, "match_success": match_success, "rating": rating, "dim": doc["dim"]},
        }


# Process result for yes_no
def tempcompass_process_results_yes_no(doc, result):
    pred = result[0]
    rating = 0
    match_success = True
    chatgpt_response = None

    yes_no_pred = extract_pred(pred)

    # Some hand-crafted matching rules
    if yes_no_pred:
        rating = 1 if yes_no_pred == doc["answer"] else 0
    else:
        match_success = False  # Fail to match answer in the video-llm response. Use ChatGPT to evaluate.
        base_prompt = """
        You will receive a Yes/No question, the ground-truth answer and the prediction from a question answering (QA) model. \
        Your task is to determine whether QA model prediction is correct, based on the question and ground-truth answer. \
        If the prediction is correct, respond "Correct". If the prediction is incorrect, respond "Incorrect".
        """
        prompt = f"""{base_prompt}\nYes/No Question:\n{doc["question"]}\nGround-Truth Answer: {doc["answer"]}\nModel Prediction: {pred}"""
        chatgpt_response, rating = get_eval_result(
            prompt,
            model=_get_gpt_eval_model(tempcompass_yes_no_config),
        )

    if chatgpt_response:
        return {
            "avg_accuracy": {
                "video_id": doc["video_id"],
                "question": doc["question"],
                "gt-answer": doc["answer"],
                "video-llm-prediction": pred,
                "match_success": match_success,
                "rating": rating,
                "chatgpt_response": chatgpt_response,
                "dim": doc["dim"],
            },
            doc["dim"]
            + "_accuracy": {
                "video_id": doc["video_id"],
                "question": doc["question"],
                "gt-answer": doc["answer"],
                "video-llm-prediction": pred,
                "match_success": match_success,
                "rating": rating,
                "chatgpt_response": chatgpt_response,
                "dim": doc["dim"],
            },
        }
    else:
        return {
            "avg_accuracy": {"video_id": doc["video_id"], "question": doc["question"], "gt-answer": doc["answer"], "video-llm-prediction": pred, "match_success": match_success, "rating": rating, "dim": doc["dim"]},
            doc["dim"] + "_accuracy": {"video_id": doc["video_id"], "question": doc["question"], "gt-answer": doc["answer"], "video-llm-prediction": pred, "match_success": match_success, "rating": rating, "dim": doc["dim"]},
        }


# Process result for caption_matching
def tempcompass_process_results_caption_matching(doc, result):
    pred = result[0]
    rating = 0
    match_success = True
    chatgpt_response = None

    eval_rule_rating = eval_rule(pred, doc["question"], doc["answer"])

    # Some hand-crafted matching rules
    if eval_rule_rating != "fail":
        rating = eval_rule_rating
    else:
        match_success = False  # Fail to match answer in the video-llm response. Use ChatGPT to evaluate.
        base_prompt = """
        You will receive a caption matching question, the ground-truth answer and the prediction from a question answering (QA) model. \
        Your task is to determine whether QA model prediction is correct, based on the question and ground-truth answer. \
        If the prediction is correct, respond "Correct". If the prediction is incorrect, respond "Incorrect".
        """
        prompt = f"""{base_prompt}\nCaption Matching Question:\n{doc["question"]}\nGround-Truth Answer: {doc["answer"]}\nModel Prediction: {pred}"""
        chatgpt_response, rating = get_eval_result(
            prompt,
            model=_get_gpt_eval_model(tempcompass_caption_matching_config),
        )

    if chatgpt_response:
        return {
            "avg_accuracy": {
                "video_id": doc["video_id"],
                "question": doc["question"],
                "gt-answer": doc["answer"],
                "video-llm-prediction": pred,
                "match_success": match_success,
                "rating": rating,
                "chatgpt_response": chatgpt_response,
                "dim": doc["dim"],
            },
            doc["dim"]
            + "_accuracy": {
                "video_id": doc["video_id"],
                "question": doc["question"],
                "gt-answer": doc["answer"],
                "video-llm-prediction": pred,
                "match_success": match_success,
                "rating": rating,
                "chatgpt_response": chatgpt_response,
                "dim": doc["dim"],
            },
        }
    else:
        return {
            "avg_accuracy": {"video_id": doc["video_id"], "question": doc["question"], "gt-answer": doc["answer"], "video-llm-prediction": pred, "match_success": match_success, "rating": rating, "dim": doc["dim"]},
            doc["dim"] + "_accuracy": {"video_id": doc["video_id"], "question": doc["question"], "gt-answer": doc["answer"], "video-llm-prediction": pred, "match_success": match_success, "rating": rating, "dim": doc["dim"]},
        }


# Process result for captioning
def tempcompass_process_results_captioning(doc, result):
    pred = result[0]

    caption_evaluation_prompt = """
    You will receive a video description and a multi-choice question. Your task is to choose the correct answer and briefly explain the reason why you choose the answer. \
    If none of the choice candidates are correct or the video description lacks enough information to answer the question, just answer "None of the choices are correct". \
    Please organize your response in this format:
    ```
    Reasoning: [Your reason to obtain the answer]
    Answer: [Your answer]
    ```

    Here are some examples of video description, multi-choice question and the expected answer:
    ```
    Video Description: A person is palying football.
    Multi-Choice Question:
    What is the person doing in the video?
    A. cooking
    B. palying football
    C. playing basketball
    D. reading book
    Reasoning: The video description mentions that the person is playing football.
    Answer: B. palying football

    Video Description: A bird is flying clockwise.
    Multi-Choice Question:
    In which direction is the bird flying?
    A. backwark
    B. counter-clockwise
    C. clockwise
    D. downward
    Reasoning: The video description mentions that the bird is flying clockwise
    Answer: C. clockwise

    Video Description: An air balloon is inflating.
    Multi-Choice Question:
    What is happening to the air balloon?
    A. exploding
    B. getting smaller
    C. flying
    Reasoning: The video description mentions that the air balloon is inflating, while none of the coices can be explained as inflating.
    Answer: None of the choices are correct
    ```
    """

    prompt = f"""{caption_evaluation_prompt}\nVideo Description:{pred}\nMulti-Choice Question:\n{doc["mc_question"]}\nAnswer:"""
    eval_result = get_eval_result_for_captioning(
        prompt,
        mc_answer=doc["mc_answer"],
        model=_get_gpt_eval_model(tempcompass_captioning_config),
    )

    return {
        "avg_accuracy": {
            "video_id": doc["video_id"],
            "question": doc["question"],
            "chatgpt-reasoning": eval_result["chatgpt-reasoning"],
            "chatgpt-answer": eval_result["chatgpt-answer"],
            "video-llm-prediction": pred,
            "gt-answer": doc["mc_answer"],
            "rating": eval_result["rating"],
            "dim": doc["dim"],
        },
        doc["dim"]
        + "_accuracy": {
            "video_id": doc["video_id"],
            "question": doc["question"],
            "chatgpt-reasoning": eval_result["chatgpt-reasoning"],
            "chatgpt-answer": eval_result["chatgpt-answer"],
            "video-llm-prediction": pred,
            "gt-answer": doc["mc_answer"],
            "rating": eval_result["rating"],
            "dim": doc["dim"],
        },
    }


# utils functions for captioning: parse gpt outputs
def parse_llm_output_for_captioning(llm_output, gt_answer):
    if llm_output == "invalid_request_error" or not llm_output:
        eval_result = {"rating": -1, "chatgpt-answer": None, "chatgpt-reasoning": None}
        return eval_result

    eval_result = {}
    lines = llm_output.split("\n")

    for line in lines:
        line = line.strip()
        if "Reasoning" in line:
            eval_result["chatgpt-reasoning"] = line.replace("Reasoning:", "").strip()
        if "Answer" in line:
            eval_result["chatgpt-answer"] = line.replace("Answer:", "").strip()

    if not "chatgpt-answer" in eval_result:
        eval_result["chatgpt-answer"] = llm_output
    if not "chatgpt-reasoning" in eval_result:
        eval_result["chatgpt-reasoning"] = None

    # Check if the chatgpt answer is the ground-truth answer
    answer_counts = sum(eval_result["chatgpt-answer"].count(prefix) for prefix in ["A.", "B.", "C.", "D."])  # calculate the number of 'A.', 'B.', 'C.', 'D.' in chatgpt-answer

    if eval_result["chatgpt-answer"].split(". ")[0] == gt_answer.split(". ")[0] and answer_counts == 1:
        eval_result["rating"] = 1
    else:
        eval_result["rating"] = 0
    return eval_result

def get_llm_output_for_captioning(prompt, model):
    messages = [
        {"role": "system", "content": "You are an AI assistant for question answering."},
        {"role": "user", "content": prompt},
    ]

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=1.0,
            max_tokens=128,
            top_p=1,
            presence_penalty=1,
        )

        llm_output = _chat_completion_content(response)

        if getattr(response, "usage", None) is not None:
            token_count = {
                "prompt_tokens": getattr(response.usage, "prompt_tokens", None),
                "completion_tokens": getattr(response.usage, "completion_tokens", None),
                "total_tokens": getattr(response.usage, "total_tokens", None),
            }
        else:
            token_count = None

        return llm_output, token_count

    except Exception as e:
        err_type = None
        if hasattr(e, "type"):
            err_type = getattr(e, "type", None)

        if err_type is None and hasattr(e, "response") and e.response is not None:
            try:
                data = e.response.json()
                err_type = (data.get("error") or {}).get("type")
            except Exception:
                pass

        if err_type == "invalid_request_error":
            return "invalid_request_error", None

        return "", None


# utils functions for captioning: consolidate and return gpt outputs
def get_eval_result_for_captioning(prompt, mc_answer, model, maxtry=10):
    while True:
        try:
            llm_output, token_count = get_llm_output_for_captioning(prompt, model)
            eval_result = parse_llm_output_for_captioning(llm_output, gt_answer=mc_answer)
            eval_result["token_count"] = token_count
            return eval_result
        except Exception as e:
            if maxtry <= 0:
                print(f"Final error after all retries: {e}")
                eval_result = {"chatgpt-reasoning": None, "chatgpt-answer": None, "rating": -1, "token_count": None}
                return eval_result
            maxtry -= 1
            print(f"Not success! {maxtry} retries remaining... Error: {e}")
            time.sleep(random.uniform(1, 2))


# utils function for caption_matching
def eval_rule(video_llm_output, question, answer):
    # Determine whether the video llm output is correct, based on word matching rules
    option_strs = question.split("\n")[1:]  # complete option strings
    option_sents = [opt.split(": ")[1] for opt in option_strs]  # option sentence
    option_inds = [opt.split(": ")[0] for opt in option_strs] + [opt.split(": ")[0].replace("Sentence ", "").replace("Option ", "").replace("Caption ", "") for opt in option_strs]  # option index, e.g., Sentence A, Caption A, Option 1
    video_llm_pred = None
    for option_str in option_strs:
        if option_str == video_llm_output:
            video_llm_pred = option_str
    for option_sent in option_sents:
        if option_sent == video_llm_output or (") " in video_llm_output and option_sent == video_llm_output.split(") ")[1]):
            video_llm_pred = option_sent
    for option_ind in option_inds:
        if option_ind == video_llm_output or option_ind == video_llm_output.replace(".", ""):
            video_llm_pred = option_ind

    if video_llm_pred is None:
        return "fail"
    else:
        return 1 if video_llm_pred == answer or video_llm_pred == answer.split(":")[0] or video_llm_pred == answer.split(": ")[1] or video_llm_pred == answer.split(": ")[0].split()[1] else 0


# utils function for yes_no
def extract_pred(video_llm_output):
    # Extract the yes/no predction from the original video llm output
    video_llm_output = video_llm_output.lower()
    if video_llm_output.startswith("yes"):
        return "yes"
    elif video_llm_output.startswith("no"):
        return "no"
    else:
        return False


# utils function for gpt_evaluation when rule-based matching is unsuccessful
def get_eval_result(prompt, model, maxtry=10, sys_prompt=None):
    llm_output = None
    while True:
        try:
            llm_output = get_llm_output(prompt, model, sys_prompt)
            rating = llm_output_to_rating(llm_output)
            return llm_output, rating
        except Exception as e:
            if maxtry <= 0:
                print(f"Final error after all retries: {e}")
                return llm_output, 0
            maxtry -= 1
            print(f"Not success! {maxtry} retries remaining... Error: {e}")
            time.sleep(random.uniform(1, 2))


# utils function for gpt evaluation
def get_llm_output(prompt, model, sys_prompt=None, max_tokens=128):
    if sys_prompt is None:
        sys_prompt = "You are an AI assistant for question answering."

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": prompt}
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=1.0,
        max_tokens=max_tokens,
        top_p=1,
        presence_penalty=1,
    )
    llm_output = _chat_completion_content(response)
    return llm_output

# utils function that converts gpt evaluation into rating
def llm_output_to_rating(llm_output):
    assert "Correct" in llm_output or "Incorrect" in llm_output
    if llm_output.startswith("Correct"):
        rating = 1
    elif llm_output.startswith("Incorrect"):
        rating = 0
    elif ("Correct" in llm_output) and ("Incorrect" not in llm_output):
        rating = 1
    elif "Incorrect" in llm_output:
        rating = 0
    return rating


# Factory into different aggregate
def tempcompass_aggregate_rating(results, args):
    yes_count = 0

    # results is a list of dict
    for answer_dict in results:
        if answer_dict["rating"] == 1:
            yes_count += 1

    accuracy = yes_count / len(results)

    return accuracy * 100

VIDEO_TYPE = ["short", "medium", "long"]
CATEGORIES = ["Knowledge", "Film & Television", "Sports Competition", "Artistic Performance", "Life Record", "Multilingual"]

SUB_CATEGORIES = [
    "Humanity & History",
    "Literature & Art",
    "Biology & Medicine",
    "Finance & Commerce",
    "Astronomy",
    "Geography",
    "Law",
    "Life Tip",
    "Technology",
    "Animation",
    "Movie & TV Show",
    "Documentary",
    "News Report",
    "Esports",
    "Basketball",
    "Football",
    "Athletics",
    "Other Sports",
    "Stage Play",
    "Magic Show",
    "Variety Show",
    "Acrobatics",
    "Handicraft",
    "Food",
    "Fashion",
    "Daily Life",
    "Travel",
    "Pet & Animal",
    "Exercise",
    "Multilingual",
]

TASK_CATEGORIES = [
    "Temporal Perception",
    "Spatial Perception",
    "Attribute Perception",
    "Action Recognition",
    "Object Recognition",
    "OCR Problems",
    "Counting Problem",
    "Temporal Reasoning",
    "Spatial Reasoning",
    "Action Reasoning",
    "Object Reasoning",
    "Information Synopsis",
]

replace_prompt = " Please answer yes or no."

# with open(Path(__file__).parent / "_default_template_yaml", "r") as f:
#     raw_data = f.readlines()
#     safe_data = []
#     for i, line in enumerate(raw_data):
#         # remove function definition since yaml load cannot handle it
#         if "!function" not in line:
#             safe_data.append(line)

#     config = yaml.safe_load("".join(safe_data))

with open(Path(__file__).parent / "videomme.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)
base_cache_dir_videomme = yaml.safe_load("".join(safe_data))["dataset_path"]

with open(Path(__file__).parent / "videomme_long.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)
base_cache_dir_videomme_long = yaml.safe_load("".join(safe_data))["dataset_path"]

def parse_subtitle_time(time_str):
    h, m, s_ms = time_str.split(":")
    s, ms = s_ms.split(",")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def load_subtitles(subtitle_path):
    subtitles = {}
    with open(subtitle_path, "r", encoding="utf-8") as file:
        content = file.read().split("\n\n")
        for section in content:
            if section.strip():
                lines = section.split("\n")
                if len(lines) >= 3:
                    time_range = lines[1].split(" --> ")
                    start_time = parse_subtitle_time(time_range[0])
                    end_time = parse_subtitle_time(time_range[1])
                    text = " ".join(line for line in lines[2:])
                    subtitles[(start_time, end_time)] = text
    return subtitles


def convert_time_to_frame(time_in_seconds, fps):
    return int(time_in_seconds * fps)


def extract_subtitles(video_path, subtitle_path):
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    total_frame = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    subtitles = load_subtitles(subtitle_path)

    subtitle_frames = []
    for (start_time, end_time), text in subtitles.items():
        start_frame = convert_time_to_frame(start_time, fps)
        end_frame = convert_time_to_frame(end_time, fps)
        subtitle_frames.append((start_frame, end_frame, text))

    return subtitle_frames, total_frame


def videmme_process_docs_base(dataset: datasets.Dataset, type: str) -> datasets.Dataset:
    return dataset.filter(lambda x: x["duration"] == type)


videomme_process_docs_long = partial(videmme_process_docs_base, type="long")


def videomme_doc_to_visual(doc):
    cache_dir = base_cache_dir_videomme
    if doc['duration'] == "long":
        cache_dir = base_cache_dir_videomme_long
    video_path = doc["videoID"] + ".mp4"
    video_path = os.path.join(cache_dir, "data", video_path)
    if os.path.exists(video_path):
        video_path = video_path
    elif os.path.exists(video_path.replace("mp4", "MP4")):
        video_path = video_path.replace("mp4", "MP4")
    elif os.path.exists(video_path.replace("mp4", "mkv")):
        video_path = video_path.replace("mp4", "mkv")
    else:
        sys.exit(f"video path:{video_path} does not exist, please check")
    return [video_path]


def videomme_doc_to_text(doc, lmms_eval_specific_kwargs=None):

    if "format" in lmms_eval_specific_kwargs and lmms_eval_specific_kwargs["format"] == "qwen3_vl":
        return videomme_doc_to_text_qwen3vl(doc, lmms_eval_specific_kwargs)

    option_prompt = "Select the best answer to the following multiple-choice question based on the video and the subtitles. Respond with only the letter (A, B, C, or D) of the correct option."
    question = doc["question"]
    option = "\n".join([f"{opt}" for i, opt in enumerate(doc["options"])])
    question = question + "\n" + option
    post_prompt = lmms_eval_specific_kwargs["post_prompt"] if "post_prompt" in lmms_eval_specific_kwargs else "The best answer is:"
    full_prompt = option_prompt + "\n" + question + "\n" + post_prompt
    return full_prompt


def videomme_doc_to_text_qwen3vl(doc, lmms_eval_specific_kwargs=None):
    """
    Adapted from: https://github.com/open-compass/VLMEvalKit/blob/main/vlmeval/vlm/qwen3_vl/prompt.py#L93
    """
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")

    question = doc["question"]
    option = "\n".join([f"{opt}" for i, opt in enumerate(doc["options"])])

    full_prompt = f"{pre_prompt}{question}\nOptions:\n{option}\n{post_prompt}"
    return full_prompt


def videomme_doc_to_text_subtitle(doc, lmms_eval_specific_kwargs=None):
    cache_dir = base_cache_dir_videomme
    if doc['duration'] == "long":
        cache_dir = base_cache_dir_videomme_long
    video_path = doc["videoID"] + ".mp4"
    video_path = os.path.join(cache_dir, "data", video_path)
    subtitle_path = os.path.join(cache_dir, "subtitle", doc["videoID"] + ".srt")
    video_path = os.path.join(cache_dir, video_path)
    if os.path.exists(subtitle_path):  # Denote have subtitle
        subtitle = open(subtitle_path).readlines()
    else:
        subtitle = ""
    subtitles_prompt = "This video's subtitles are listed below: \n"
    if subtitle == "":
        subtitle = "No subtitles available"
    else:
        if "gemini_api_flag" in lmms_eval_specific_kwargs:  # specific for gemini_api
            if lmms_eval_specific_kwargs["gemini_api_flag"] == "full subtitle":
                textlist = []
                for ele in subtitle:
                    pattern = r'<font color="white" size=".72c">(.*?)</font>'
                    matches = re.findall(pattern, ele)
                    if matches:
                        textlist.append(matches[0])
                subtitle_text = "\n".join(textlist)
        else:
            if "frame_num" in lmms_eval_specific_kwargs:
                frame_num = lmms_eval_specific_kwargs["frame_num"]
                subtitle_by_frame, total_frame = extract_subtitles(video_path, subtitle_path)
                if frame_num == -1:
                    frame_num = total_frame
                uniform_sampled_frames = np.linspace(0, total_frame - 1, frame_num, dtype=int).tolist()

                subtitle_by_frame_idx = []
                for frame_idx in uniform_sampled_frames:
                    for idx, title in enumerate(subtitle_by_frame):
                        if frame_idx < title[1] and frame_idx >= title[0]:
                            subtitle_by_frame_idx.append(idx)
                subtitle_by_frame_idx = list(set(subtitle_by_frame_idx))

                textlist = []
                for idx in subtitle_by_frame_idx:
                    pattern = r'<font color="white" size=".72c">(.*?)</font>'
                    raw_text = re.findall(pattern, subtitle_by_frame[idx][2])
                    try:
                        textlist.append(raw_text[0])
                    except:
                        continue
                subtitle_text = "\n".join(textlist)
        subtitle = subtitle_text

    if "format" in lmms_eval_specific_kwargs and lmms_eval_specific_kwargs["format"] == "qwen3_vl":
        prompt = videomme_doc_to_text_qwen3vl(doc, lmms_eval_specific_kwargs)
        full_prompt = subtitles_prompt + subtitle + "\n" + prompt
        return full_prompt

    option_prompt = "Select the best answer to the following multiple-choice question based on the video and the subtitles. Respond with only the letter (A, B, C, or D) of the correct option."
    question = doc["question"]
    option = "\n".join([f"{opt}" for i, opt in enumerate(doc["options"])])
    question = question + "\n" + option
    full_prompt = subtitles_prompt + subtitle + "\n" + option_prompt + "\n" + question + "\n" + "The best answer is:"
    return full_prompt


def extract_characters_regex(s):
    s = s.strip()
    answer_prefixes = [
        "The best answer is",
        "The correct answer is",
        "The answer is",
        "The answer",
        "The best option is" "The correct option is",
        "Best answer:" "Best option:",
    ]
    for answer_prefix in answer_prefixes:
        s = s.replace(answer_prefix, "")

    if len(s.split()) > 10 and not re.search("[ABCD]", s):
        return ""

    matches = re.search(r"[ABCD]", s)
    if matches is None:
        return ""
    return matches[0]


matrices = []

for i in VIDEO_TYPE:
    for j in CATEGORIES:
        for k in SUB_CATEGORIES:
            for l in TASK_CATEGORIES:
                matrices.append(f"{i}_{j}_{k}_{l}")


def videomme_process_results(doc, results):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name (in this case videomme score), value: metric value
    """
    pred = results[0]
    pred_ans = extract_characters_regex(pred)
    # gt_ans = doc["answer"].lower().strip().replace(".", "")

    category = doc["domain"]
    sub_category = doc["sub_category"]
    task_category = doc["task_type"]
    data_dict = {"question_id": doc["question_id"], "duration": doc["duration"], "category": category, "sub_category": sub_category, "task_category": task_category, "pred_answer": pred_ans, "answer": doc["answer"]}

    # return {f"videomme_perception_score": data_dict for metric in matrices}
    return {f"videomme_perception_score": data_dict}


def videomme_aggregate_results(results):
    """
    Args:
        results: a list of values returned by process_results
    Returns:
        A score
    """
    category2score = {}

    for video_type in VIDEO_TYPE:
        for category in CATEGORIES:
            for sub_category in SUB_CATEGORIES:
                for task_category in TASK_CATEGORIES:
                    key = f"{video_type}_{category}_{sub_category}_{task_category}"
                    category2score[key] = {"correct": 0, "answered": 0}

    for result in results:
        video_type = result["duration"]
        category = result["category"]
        sub_category = result["sub_category"]
        task_category = result["task_category"]
        key = f"{video_type}_{category}_{sub_category}_{task_category}"
        category2score[key]["answered"] += 1
        category2score[key]["correct"] += result["pred_answer"] == result["answer"]

    for video_type in VIDEO_TYPE:
        total_correct = 0
        total_answered = 0
        for k, v in category2score.items():
            if video_type in k:
                total_correct += v["correct"]
                total_answered += v["answered"]
        eval_logger.info(f"Evaluation on video Type: {video_type}: {100 * total_correct / total_answered if total_answered > 0 else 0 : .1f}%")

    for category in CATEGORIES:
        total_correct = 0
        total_answered = 0
        for k, v in category2score.items():
            if category in k:
                total_correct += v["correct"]
                total_answered += v["answered"]
        eval_logger.info(f"Evaluation on Categories: {category}: {100 * total_correct / total_answered if total_answered > 0 else 0 : .1f}%")

    for sub_cate in SUB_CATEGORIES:
        total_correct = 0
        total_answered = 0
        for k, v in category2score.items():
            if sub_cate in k:
                total_correct += v["correct"]
                total_answered += v["answered"]
        eval_logger.info(f"Evaluation on Video Sub Categories: {sub_cate}: {100 * total_correct / total_answered if total_answered > 0 else 0 : .1f}%")

    for task_cate in TASK_CATEGORIES:
        total_correct = 0
        total_answered = 0
        for k, v in category2score.items():
            if task_cate in k:
                total_correct += v["correct"]
                total_answered += v["answered"]
        eval_logger.info(f"Evaluation on Task Categories: {task_cate}: {100 * total_correct / total_answered if total_answered > 0 else 0 : .1f}%")

    total_correct = 0
    total_answered = 0
    for k, v in category2score.items():
        total_correct += v["correct"]
        total_answered += v["answered"]
    eval_logger.info(f"Overall Performance: {100 * total_correct / total_answered if total_answered > 0 else 0 : .1f}%")
    return 100 * total_correct / total_answered if total_answered > 0 else 0


# mtcbench_video_utils.py
"""
MTC-Bench video utils for:
  - Charades-STA (temporal grounding -> R@k / IoU)
  - MotionBench (QA -> accuracy)
  - MMVU (QA -> accuracy)
  - HR-Bench (image QA -> accuracy)
  - Vstar (image QA -> accuracy)
  - CapsBench (captioning -> BLEU/CIDEr etc.; prefer pycocoevalcap if installed)

Usage in lmms-eval YAML:
  doc_to_visual: !function mtcbench_video_utils.charades_doc_to_visual
  process_results: !function mtcbench_video_utils.generic_process_results
  metric_list:
    - metric: R_at_K_IoU
      aggregation: !function mtcbench_video_utils.charades_r_at_k_aggregate
    - metric: ACCURACY
      aggregation: !function mtcbench_video_utils.accuracy_aggregate_results
    - metric: BLEU-4
      aggregation: !function mtcbench_video_utils.caps_caption_aggregate
"""

def _first_present(d: Dict, keys: List[str], default=None):
    """Return first value from dict d matching any key in keys."""
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def normalize_answer(s):
    """Normalize answer string for comparison."""
    if s is None:
        return ""
    return str(s).strip().lower()


def parse_intervals(text) -> List[Tuple[float, float]]:
    """Parse interval strings like '10.5-20.3' into (start, end) tuples."""
    if text is None:
        return []
    s = str(text).strip()
    if not s:
        return []
    # Handle list/tuple format
    if isinstance(text, (list, tuple)):
        if len(text) >= 2:
            try:
                return [(float(text[0]), float(text[1]))]
            except (ValueError, TypeError):
                return []
        return []
    # Parse string format: "10.5-20.3" or "10.5,20.3"
    for sep in ['-', ',']:
        if sep in s:
            parts = s.split(sep)
            if len(parts) >= 2:
                try:
                    return [(float(parts[0].strip()), float(parts[1].strip()))]
                except ValueError:
                    pass
    return []


def interval_iou(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Compute IoU between two intervals."""
    inter_start = max(a[0], b[0])
    inter_end = min(a[1], b[1])
    inter_len = max(0, inter_end - inter_start)
    union_len = max(a[1], b[1]) - min(a[0], b[0])
    return inter_len / union_len if union_len > 0 else 0.0

def accuracy_mean_results(eval_items: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = 0
    total_score = 0.0
    for it in eval_items:
        total_score += float(it)
        n += 1
    acc = total_score / n if n > 0 else 0.0
    return acc

# -------------------- HR-Bench (image QA) --------------------
with open(Path(__file__).parent / "hr_bench.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)

    hrbench_config = yaml.safe_load("".join(safe_data))

hrbench_evaluator = HRBenchEval(api_key=os.getenv("OPENAI_API_KEY", "API_KEY"), gpt_model=_get_gpt_eval_model(hrbench_config), max_workers=hrbench_config["metadata"]["max_workers"])


def decode_base64_to_image(base64_string, target_size=-1):
    image_data = base64.b64decode(base64_string)
    image = Image.open(BytesIO(image_data)).convert("RGB")
    if target_size > 0:
        image.thumbnail((target_size, target_size))
    return image


def hrbench_doc_to_visual(doc):
    image = decode_base64_to_image(doc["image"])
    return [image]


def hrbench_doc_to_options(doc):
    options = {cand: doc[cand] for cand in string.ascii_uppercase if cand in doc and not pd.isna(doc[cand])}
    return options


def hrbench_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["question"].strip()
    options = hrbench_doc_to_options(doc)
    options_prompt = ""
    for key, item in options.items():
        options_prompt += f"{key}. {item}\n"
    prompt = ""
    prompt += f"{question}\n{options_prompt}Answer the option letter directly."
    return prompt


def hrbench_process_results(doc, results):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name, value: metric value
    """
    pred = results[0].strip()
    gt = doc["answer"]
    options = hrbench_doc_to_options(doc)
    question = doc["question"]
    resp_dic = hrbench_evaluator.get_chat_response({"question": question, "options": options, "prediction": pred})
    gpt_prediction = resp_dic["gpt_prediction"]
    category = doc["category"]
    cycle_category = doc["cycle_category"]

    gpt_score = 0
    if gt.lower() == gpt_prediction.lower():
        gpt_score = 1

    return {category: {"index": doc["index"], "cycle_category": cycle_category, "gpt_score": gpt_score}, "average": {"index": doc["index"], "cycle_category": cycle_category, "gpt_score": gpt_score}}


def hrbench_aggregate_results(results, args):
    """
    Args:
        results: a list of values returned by process_results
    Returns:
        A score
    """
    cycle_category_scores = defaultdict(list)
    for result in results:
        score = result["gpt_score"]
        cycle_category = result["cycle_category"]
        cycle_category_scores[cycle_category].append(score)

    cycle_category_avg_score = {}
    for cycle_category, scores in cycle_category_scores.items():
        avg_score = sum(scores) / len(scores)
        cycle_category_avg_score[cycle_category] = avg_score

    avg_score = sum(cycle_category_avg_score.values()) / len(cycle_category_avg_score)
    return avg_score

# -------------------- Vstar --------------------
def vstar_doc_to_visual(doc):
    """Return image for Vstar task."""
    img = _first_present(doc, ["image", "img", "image_bytes"])
    if img is None:
        return []
    # Convert to RGB if needed
    if hasattr(img, "convert"):
        return [img.convert("RGB")]
    return [img]


def vstar_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    """Return question text for Vstar task."""
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    text = str(_first_present(doc, ["text", "question", "prompt"], ""))
    return f"{pre_prompt}{text}{post_prompt}"


def vstar_process_results(doc, results):
    """Process results for Vstar task."""
    assert len(results) == 1, f"Expected 1 result, got {len(results)}"
    pred = results[0]
    
    # Get ground truth
    ans = _first_present(doc, ["label", "answer", "gt"])
    if ans is None:
        targets = []
    else:
        targets = [str(ans)]
    
    # Compute accuracy
    pred_norm = normalize_answer(pred)
    targets_norm = [normalize_answer(t) for t in targets]
    accuracy = 1.0 if pred_norm in targets_norm else 0.0
    
    return {
        "exact_match": accuracy,
        "submission": {
            "prediction": pred,
            "answer": targets[0] if targets else ""
        }
    }

# -------------------- Charades-STA (Captioning) --------------------
with open(Path(__file__).parent / "charades.yaml", "r") as f:
    raw_data = f.readlines()
safe_data = []
for line in raw_data:
    if "!function" not in line:
        safe_data.append(line)

_cfg = yaml.safe_load("".join(safe_data))
base_cache_dir_charades = _cfg["dataset_path"]


# -----------------------------
# Data mapping functions
# -----------------------------
def temporal_grounding_doc_to_visual(doc, lmms_eval_specific_kwargs=None):
    video_path = doc["video"]
    video_path = os.path.join(base_cache_dir_charades, "Charades_v1_480", video_path)

    if os.path.exists(video_path):
        return [video_path]
    if "s3://" not in video_path:
        sys.exit(f"video path:{video_path} does not exist, please check")
    return [video_path]


def temporal_grounding_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}

    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")

    question = doc["caption"]
    return f"{pre_prompt}{question}. {post_prompt}"


def temporal_grounding_doc_to_answer(doc):
    # e.g. "[12.3, 18.7]" or "[[12.3, 18.7]]" depending on dataset
    return doc["timestamp"]


# -----------------------------
# Time parsing + IoU
# -----------------------------
def extract_time(paragraph: str):
    """
    Parse model output text and extract a single [start, end] in seconds.
    Returns: list like [[start, end]] or [] if cannot parse.
    """
    if not paragraph:
        return []

    prompt = "A specific example is : 20.8 - 30.0 seconds".lower()
    paragraph = paragraph.lower().replace(prompt, "").replace("to", "-")

    # Split text into sentences based on common delimiters
    sentences = re.split(r"[!?\n]", paragraph)

    keywords = ["starts", "ends", "happens in", "start time", "end time", "start", "end", "happen"]
    candidates = [s for s in sentences if any(k in s for k in keywords)]
    if len(candidates) == 0:
        candidates = sentences

    timestamps = []

    main_pattern = re.compile(r"(\d+(?:\.\d+)?)\s*[–-]\s*(\d+(?:\.\d+)?)")
    time_number_pattern = re.compile(r"\b(\d+(?:\.\d+)?)\b")
    time_format_pattern = re.compile(r"\b(\d{1,2}:\d{2}:\d{2})\b")
    fallback_pattern = re.compile(r"(\d+(?:\.\d+)?)\s*s?\s*[–-]\s*(\d+(?:\.\d+)?)\s*s?")

    for sentence in candidates:
        time_matches = main_pattern.findall(sentence)
        if time_matches:
            timestamps = [[float(a), float(b)] for a, b in time_matches]
            break

    if len(timestamps) == 0:
        times = []
        for sentence in candidates:
            nums = time_number_pattern.findall(sentence)
            if nums:
                for n in nums:
                    try:
                        times.append(float(n))
                    except Exception:
                        pass
        times = times[: len(times) // 2 * 2]
        if len(times) >= 2:
            timestamps = [(times[i], times[i + 1]) for i in range(0, len(times), 2)]

    if len(timestamps) == 0:
        times = []
        for sentence in candidates:
            ts = time_format_pattern.findall(sentence)
            for t in ts:
                # convert to seconds
                parts = t.split(":")
                if len(parts) == 3:
                    h, m, s = map(int, parts)
                    times.append(h * 3600 + m * 60 + s)
        times = times[: len(times) // 2 * 2]
        if len(times) >= 2:
            timestamps = [(times[i], times[i + 1]) for i in range(0, len(times), 2)]

    if len(timestamps) == 0:
        for sentence in candidates:
            matches = fallback_pattern.findall(sentence)
            if matches:
                timestamps = [[float(a), float(b)] for a, b in matches]
                break

    results = []
    for start, end in timestamps:
        if end >= start:
            results.append([start, end])
        else:
            results.append([end, start])

    if len(results) > 1:
        results = results[:1]
    return results


def iou(interval_a, interval_b):
    """
    interval: [start, end]
    """
    a0, a1 = interval_a
    b0, b1 = interval_b
    inter = max(min(a1, b1) - max(a0, b0), 0.0)
    union = max(a1, b1) - min(a0, b0)
    if union <= 0:
        return 0.0
    return inter / union


def _normalize_gt_timestamp(gt):
    """
    dataset may store gt as:
      - [s, e]
      - [[s, e]]
      - string repr like "[s, e]" or "[[s, e]]"
    Return [s, e] float list.
    """
    if isinstance(gt, str):
        gt = eval(gt)

    if isinstance(gt, (list, tuple)) and len(gt) == 1 and isinstance(gt[0], (list, tuple)) and len(gt[0]) == 2:
        gt = gt[0]

    if not (isinstance(gt, (list, tuple)) and len(gt) == 2):
        raise ValueError(f"Unexpected gt format: {gt}")

    return [float(gt[0]), float(gt[1])]


def _safe_pred_interval(pred_text, gt_interval):
    """
    If cannot parse, follow your old behavior: assign an out-of-range interval
    so IoU becomes 0 in most cases.
    """
    ts = extract_time(pred_text)
    if len(ts) != 1:
        # push away from gt; old code used gt[1]+10 to gt[1]+20
        return [gt_interval[1] + 10.0, gt_interval[1] + 20.0]
    return ts[0]

def temporal_grounding_process_results_charades_iou(doc, result):
    """
    result: list[str] where result[0] is generated text
    Must return dict of metric_name -> per-sample value
    """
    pred_text = result[0] if isinstance(result, (list, tuple)) and len(result) > 0 else ""
    gt_interval = _normalize_gt_timestamp(doc["timestamp"])
    pred_interval = _safe_pred_interval(pred_text, gt_interval)

    cur_iou = iou(gt_interval, pred_interval)
    key = f'{doc["video"]}>>>{doc["caption"]}>>>{doc["timestamp"]}'
    submission = {key: pred_text}

    return {
        "submission": submission,
        "miou": float(cur_iou),
        "iou@0.3": 1.0 if cur_iou >= 0.3 else 0.0,
        "iou@0.5": 1.0 if cur_iou >= 0.5 else 0.0,
        "iou@0.7": 1.0 if cur_iou >= 0.7 else 0.0,
    }


def temporal_grounding_aggregate_mean(results, args):
    """
    results: list of per-sample floats (already extracted by lmms-eval for this metric)
    Return a single float.
    """
    if results is None or len(results) == 0:
        return 0.0
    return sum(results) / len(results)


def temporal_grounding_aggregate_charades(results, args):
    temporal_grounding_aggregate_submissions(results, args, "charades")


def temporal_grounding_aggregate_submissions(results, args, task):
    now_date_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    submission_file_name = f"inference_results_temporal_grounding_{task}_{now_date_time}.json"
    path = file_utils.generate_submission_file(submission_file_name, args)

    combined_submission = {}
    for submission_dict in results:
        combined_submission.update(submission_dict)

    with open(path, "w") as f:
        json.dump(combined_submission, f, indent=4)

    eval_logger.info(f"Submission file saved to {path}")
    return path


# -------------------- MotionBench (QA) --------------------
with open(Path(__file__).parent / "motionbench.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for line in raw_data:
        if "!function" not in line:
            safe_data.append(line)
motionbench_config = yaml.safe_load("".join(safe_data))
motionbench_dataset_dir = motionbench_config["dataset_path"]

def _motionbench_resolve_video_path(video_path: str) -> str:
    if not video_path:
        return ""
    candidates = [
        video_path,
        os.path.join(motionbench_dataset_dir, video_path),
        os.path.join(motionbench_dataset_dir, "video", video_path),
    ]
    for cand in candidates:
        if cand and os.path.exists(cand):
            return cand
    return video_path

def motionbench_doc_to_visual(doc):
    """Return video path for MotionBench task."""
    video = _first_present(doc, ["video_path", "video", "file"])
    if not video:
        return []
    video_path = _motionbench_resolve_video_path(video)
    if os.path.exists(video_path):
        return [video_path]
    if isinstance(video_path, str) and video_path.startswith("s3://"):
        return [video_path]
    sys.exit(f"video path:{video_path} does not exist, please check")


def motionbench_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    """Return question text for MotionBench task."""
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")

    qa = _first_present(doc, ["qa"]) or []
    if isinstance(qa, list) and qa:
        text = str(_first_present(qa[0], ["question", "q"], ""))
    else:
        text = str(_first_present(doc, ["question", "caption", "text"], ""))

    return f"{pre_prompt}{text}{post_prompt}"


def motionbench_doc_to_target(doc):
    qa = _first_present(doc, ["qa"]) or []
    if isinstance(qa, list) and qa:
        return str(_first_present(qa[0], ["answer", "label", "ans"], "")).strip()
    return str(_first_present(doc, ["answer", "label"], "")).strip()


def _motionbench_extract_choice(prediction: str) -> str:
    if prediction is None:
        return ""
    text = str(prediction).strip()
    if len(text) == 1 and text.upper() in "ABCDE":
        return text.upper()
    upper = text.upper()
    match = re.search(r"\b([A-E])\b", upper)
    if match:
        return match.group(1)
    match = re.search(r"([A-E])\s*[\).:]", upper)
    if match:
        return match.group(1)
    return ""


def motionbench_process_results(doc, results):
    """Process results for MotionBench QA task."""
    assert len(results) == 1, f"Expected 1 result, got {len(results)}"
    pred = results[0]
    if isinstance(pred, list):
        pred = pred[0] if pred else ""

    answer = motionbench_doc_to_target(doc)
    if not answer or answer.upper() == "NA":
        accuracy = 0.0
    else:
        pred_choice = _motionbench_extract_choice(pred)
        accuracy = 1.0 if pred_choice.upper() == answer.upper() else 0.0

    return {"accuracy": {"exact_match": accuracy}}


# -------------------- MMVU --------------------
with open(Path(__file__).parent / "mmvu_val.yaml", "r") as f:
    raw_data_val = f.readlines()
    safe_data_val = []
    for i, line in enumerate(raw_data_val):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data_val.append(line)
mmvu_config = yaml.safe_load("".join(safe_data_val))
cache_dir_val = mmvu_config["dataset_path"]

def mmvu_doc_to_visual_val(doc):
    video_path = doc["video"]
    video_path = os.path.join(cache_dir_val, video_path)
    if os.path.exists(video_path):
        video_path = video_path
    else:
        sys.exit(f"video path:{video_path} does not exist, please check")
    return [video_path]


multiple_choice_prompt = """
    Question:{question}
    A: {a}
    B: {b}
    C: {c}
    D: {d}
    E: {e}
    Visual Information: processed video
    Do not generate any intermediate reasoning process. Answer directly with the option letter from the
    given choices.
    """

open_ended_prompt = """
    Question:{question}
    Visual Information: processed video
    Do not generate any intermediate reasoning process. Directly output the final answer.
    """

multiple_choice_prompt_cot = """
    Question:{question}
    A: {a}
    B: {b}
    C: {c}
    D: {d}
    E: {e}
    Visual Information: processed video
    Answer the given multiple-choice question step by step. Begin by explaining your reasoning process
    clearly. Conclude by stating the final answer using the following format: "Therefore, the final answer
    is: $LETTER" (without quotes), where $LETTER is one of the options. Think step by step before
    answering.
    """

open_ended_prompt_cot = """
    Question:{question}
    Visual Information: processed video
    Answer the given question step by step. Begin by explaining your reasoning process clearly. Conclude
    by stating the final answer using the following format: "Therefore, the final answer is: "Answer:
    $ANSWER" (without quotes), where $ANSWER is the final answer of the question. Think step by
    step before answering.
    """


def mmvu_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question_type = doc["question_type"]
    if question_type == "multiple-choice":
        question = doc["question"]
        choices = doc["choices"]
        full_prompt = multiple_choice_prompt.format(question=question, a=choices["A"], b=choices["B"], c=choices["C"], d=choices["D"], e=choices["E"])
    else:
        question = doc["question"]
        full_prompt = open_ended_prompt.format(question=question)
    return full_prompt


def mmvu_doc_to_text_cot(doc, lmms_eval_specific_kwargs=None):
    question_type = doc["question_type"]
    if question_type == "multiple-choice":
        question = doc["question"]
        choices = doc["choices"]
        full_prompt = multiple_choice_prompt_cot.format(question=question, a=choices["A"], b=choices["B"], c=choices["C"], d=choices["D"], e=choices["E"])
    else:
        question = doc["question"]
        full_prompt = open_ended_prompt_cot.format(question=question)
    return full_prompt


def construct_question_prompt(doc):
    """Construct the question prompt for evaluation"""
    question_type = doc["question_type"]
    if question_type == "multiple-choice":
        choices = doc["choices"]
        return f"""Question: {doc["question"]}
            A: {choices["A"]}
            B: {choices["B"]}
            C: {choices["C"]}
            D: {choices["D"]}
            E: {choices["E"]}"""
    else:
        return f"Question: {doc['question']}"


def normalize_math_notation(text: str) -> str:
    """Normalize mathematical notation for comparison (e.g., n² -> n^2, n³ -> n^3)"""
    # Convert superscript numbers to caret notation
    superscript_map = {"²": "^2", "³": "^3", "¹": "^1", "⁰": "^0", "⁴": "^4", "⁵": "^5", "⁶": "^6", "⁷": "^7", "⁸": "^8", "⁹": "^9"}
    normalized = text
    for sup, caret in superscript_map.items():
        normalized = normalized.replace(sup, caret)
    return normalized


def evaluate_with_rule_based(doc: Dict[str, Any], prediction: str) -> bool:
    """Rule-based evaluation - returns True if correct, False otherwise"""
    answer = doc["answer"]
    question_type = doc["question_type"]

    if question_type == "multiple-choice":
        # Rule-based evaluation for multiple-choice questions
        pred_str = str(prediction).strip()
        answer_str = str(answer).strip()

        # Method 1: Extract letter from prediction and compare
        letter_match = re.search(r"\b([A-E])\b", pred_str, re.IGNORECASE)
        if letter_match:
            extracted_letter = letter_match.group(1).upper()
            if extracted_letter == answer_str.upper():
                return True

        # Method 2: Check if answer letter appears in prediction (case-insensitive)
        if answer_str.upper() in pred_str.upper():
            # Make sure it's a standalone letter, not part of another word
            if re.search(rf"\b{re.escape(answer_str)}\b", pred_str, re.IGNORECASE):
                return True

        # Method 3: Check if the full answer text (letter + content) matches
        if answer_str in doc.get("choices", {}):
            choice_text = doc["choices"][answer_str].strip().lower()
            pred_lower = pred_str.lower()
            # Check if choice text appears in prediction
            if choice_text in pred_lower:
                return True
            # Check if prediction contains both the letter and key words from choice
            if answer_str.upper() in pred_str.upper():
                # Extract key words from choice (remove common words)
                words = [w.strip(string.punctuation) for w in choice_text.split() if len(w.strip(string.punctuation)) > 2]
                if len(words) > 0:
                    # Check if at least one key word appears in prediction
                    if any(word in pred_lower for word in words):
                        return True

        return False
    else:
        # Rule-based evaluation for open-ended questions
        pred_normalized = str(prediction).strip().lower()
        answer_normalized = str(answer).strip().lower()

        # Normalize mathematical notation (e.g., n² -> n^2)
        pred_normalized = normalize_math_notation(pred_normalized)
        answer_normalized = normalize_math_notation(answer_normalized)

        # Remove common punctuation and extra whitespace for comparison
        pred_clean = " ".join(pred_normalized.split())
        answer_clean = " ".join(answer_normalized.split())

        # Method 1: Exact match (after normalization)
        if pred_clean == answer_clean:
            return True

        # Method 2: Check if answer appears as a substring in prediction
        # This handles cases like: answer="Depth-First Search (DFS)", prediction="The algorithm is depth-first search (DFS)."
        if answer_clean in pred_clean:
            return True

        # Method 3: Check if prediction appears as a substring in answer (for shorter predictions)
        if pred_clean in answer_clean:
            return True

        # Method 4: For numerical answers, try to extract and compare numbers
        # Extract numbers from both strings
        pred_numbers = re.findall(r"\d+\.?\d*", pred_normalized)
        answer_numbers = re.findall(r"\d+\.?\d*", answer_normalized)

        if len(answer_numbers) > 0 and len(pred_numbers) > 0:
            # Try to match numbers (allowing for floating point differences)
            try:
                answer_num = float(answer_numbers[0])
                pred_num = float(pred_numbers[0])
                # Allow small floating point differences (0.01 tolerance)
                if abs(answer_num - pred_num) < 0.01:
                    return True
            except ValueError:
                pass

        # Method 5: Word-level matching for short answers (2-5 words)
        answer_words = [w.strip(string.punctuation) for w in answer_clean.split() if w.strip(string.punctuation)]
        pred_words = [w.strip(string.punctuation) for w in pred_clean.split() if w.strip(string.punctuation)]

        if 2 <= len(answer_words) <= 5:
            # Check if all answer words appear in prediction (order-independent)
            if all(word in pred_words for word in answer_words if len(word) > 2):
                return True

        # Method 6: Special handling for mathematical complexity notation (O(n²) vs O(n^2))
        # Extract O() notation patterns
        pred_o_match = re.search(r"O\s*\([^)]+\)", pred_clean, re.IGNORECASE)
        answer_o_match = re.search(r"O\s*\([^)]+\)", answer_clean, re.IGNORECASE)
        if pred_o_match and answer_o_match:
            pred_o_content = normalize_math_notation(pred_o_match.group(0).lower())
            answer_o_content = normalize_math_notation(answer_o_match.group(0).lower())
            if pred_o_content == answer_o_content:
                return True

        return False

def _parse_binary_judge_output(text: str) -> str:
    """
    Parse judge output and return "1" or "0" or "" if invalid.
    We enforce a strict binary protocol.
    """
    if not text:
        return ""
    s = text.strip()
    first = s[0]
    if first in ("0", "1"):
        return first
    for ch in s:
        if ch in ("0", "1"):
            return ch
    return ""

def evaluate_with_llm_judge(doc: Dict[str, Any], prediction: str) -> Tuple[bool, str]:
    """
    Hybrid evaluation:
    1) rule-based first
    2) if rule-based False:
       - multiple-choice: do NOT use GPT
       - open-ended: use GPT judge for semantic equivalence
    Returns: (is_correct, evaluation_method) where evaluation_method is "rule-based" or "gpt-based"
    """
    rule_correct = evaluate_with_rule_based(doc, prediction)
    if rule_correct:
        return True, "rule-based"

    question_type = doc.get("question_type", "")
    if question_type == "multiple-choice":
        return False, "rule-based"

    try:
        model_version = _get_gpt_eval_model(mmvu_config)

        formatted_question = construct_question_prompt(doc)
        full_answer = str(doc.get("answer", ""))

        custom_prompt = (
            "You are a strict evaluator assessing answer correctness. "
            "You must output 1 for fully correct answers and 0 for any other case.\n\n"
            "# Evaluation Rules for Open-Ended Questions\n"
            "- The model prediction may contain reasoning; focus on extracting the final answer.\n"
            "- Score 1 if the prediction matches the answer semantically, even if in different format.\n"
            "- For mathematical notation, treat equivalent forms as correct (e.g., O(n²) = O(n^2), n² = n^2).\n"
            "- Score 0 for partially correct answers or answers with extra incorrect information.\n"
            "- Ignore minor differences in formatting, capitalization, or spacing.\n"
            "- Treat numerical answers as correct if they match within reasonable precision.\n"
            "- For questions requiring units, both value and unit must be correct.\n\n"
            'Return only "1" or "0" with no additional text or formatting.'
        )

        user_content = (
            f"Question:\n{formatted_question}\n\n"
            f"Ground-truth Answer:\n{full_answer}\n\n"
            f"Model Prediction:\n{prediction}\n\n"
            "Output:"
        )

        response = client.chat.completions.create(
            model=model_version,
            messages=[
                {"role": "system", "content": custom_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=0,
            max_tokens=8,
        )

        content = (response.choices[0].message.content or "").strip()
        judge_score = _parse_binary_judge_output(content)

        if judge_score == "1":
            return True, "gpt-based"
        if judge_score == "0":
            return False, "gpt-based"

        eval_logger.error(f"GPT judge returned invalid output: {content!r}")
        return rule_correct, "rule-based"

    except Exception as e:
        eval_logger.error(f"Error getting GPT judge response: {e}")
        return rule_correct, "rule-based"


def extract_category(doc):
    category = doc["video"].split("/")[-2]
    return category


def mmvu_process_results(doc, results):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name (in this case videomme score), value: metric value
    """
    # Handle the case where results[0] might be a list or a string
    pred = results[0]
    if isinstance(pred, list):
        pred_ans = pred[0] if pred else ""
    else:
        pred_ans = pred

    # Ensure pred_ans is a string
    pred_ans = str(pred_ans)

    category = extract_category(doc)

    # Use hybrid evaluation: rule-based first, then GPT if needed
    correct, eval_method = evaluate_with_llm_judge(doc, pred_ans)

    # Extract predicted answer for logging (best effort)
    if doc["question_type"] == "multiple-choice":
        # Try to extract the letter choice from the prediction
        letter_match = re.search(r"\b([A-E])\b", pred_ans)
        extracted_answer = letter_match.group(1) if letter_match else "N/A"
    else:
        # For open-ended, just use the prediction as-is (truncated for logging)
        extracted_answer = pred_ans[:100] + "..." if len(pred_ans) > 100 else pred_ans

    data_dict = {"question_id": doc["id"], "category": category, "pred_answer": extracted_answer, "answer": doc["answer"], "correct": int(correct), "eval_method": eval_method}  # "rule-based" or "gpt-based"

    return {f"accuracy": data_dict}


def mmvu_aggregate_results_val(results):
    """
    Args:
        results: a list of values returned by process_results
    Returns:
        A score
    """

    TASK_MAP = {
        "Biology": "Science",
        "Chemistry": "Science",
        "Modern_Physics": "Science",
        "Astronomy": "Science",
        "Geography": "Science",
        "Materials_Science": "Science",
        "Neurobiology": "Science",
        "Electromagnetism": "Science",
        "Thermodynamics": "Science",
        "Mechanics": "Science",
        "Civil_Engineering": "Engineering",
        "Electrical_Engineering": "Engineering",
        "Mechanical_Engineering": "Engineering",
        "Biomedical_Engineering": "Engineering",
        "Electronics_and_Communication": "Engineering",
        "Computer_Science": "Engineering",
        "Clinical_Medicine": "Healthcare",
        "Basic_Medicine": "Healthcare",
        "Preventive_Medicine": "Healthcare",
        "Pharmacy": "Healthcare",
        "Dentistry": "Healthcare",
        "Art": "Humanities_and_Social_Science",
        "Literature": "Humanities_and_Social_Science",
        "History": "Humanities_and_Social_Science",
        "Law": "Humanities_and_Social_Science",
        "Economics": "Humanities_and_Social_Science",
        "Management": "Humanities_and_Social_Science",
    }

    TASK_TYPES = list(set(TASK_MAP.values()))

    category2score = {}
    for task_type in TASK_TYPES:
        category2score[task_type] = {"correct": 0, "answered": 0}

    for result in results:
        category = result["category"]
        if category in TASK_MAP:
            category = TASK_MAP[category]
            category2score[category]["answered"] += 1
            category2score[category]["correct"] += result.get("correct", False)
    category_scores = {}

    for category in TASK_TYPES:
        total_correct = category2score[category]["correct"]
        total_answered = category2score[category]["answered"]
        accuracy = total_correct / total_answered if total_answered > 0 else 0
        category_scores[category] = accuracy

    total_correct = sum(category2score[category]["correct"] for category in TASK_TYPES)
    total_answered = sum(category2score[category]["answered"] for category in TASK_TYPES)
    accuracy = total_correct / total_answered if total_answered > 0 else 0
    eval_logger.info("=" * 50)
    eval_logger.info(f"Average Accuracy: {accuracy:.2f}%")
    eval_logger.info("Categorical accuracy: ")
    for key, value in category_scores.items():
        eval_logger.info(f"{key} accuracy: {value:.2f}%")
    eval_logger.info("=" * 50)
    return accuracy


# -------------------- ActivityNet Captions (captioning) --------------------
with open(Path(__file__).parent / "activitynet_captions.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for line in raw_data:
        if "!function" not in line:
            safe_data.append(line)
activitynet_captions_config = yaml.safe_load("".join(safe_data))
activitynet_captions_dataset_dir = activitynet_captions_config["dataset_path"]

def activitynet_captions_doc_to_visual(doc):
    video = _first_present(doc, ["video", "video_path", "file"])
    if not video:
        return []
    candidates = [
        video,
        os.path.join(activitynet_captions_dataset_dir, video),
        os.path.join(activitynet_captions_dataset_dir, "Activity_Videos", video),
    ]
    for cand in candidates:
        if cand and os.path.exists(cand):
            return [cand]
    if isinstance(video, str) and video.startswith("s3://"):
        return [video]
    sys.exit(f"video path:{video} does not exist, please check")


def activitynet_captions_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    text = str(_first_present(doc, ["prompt", "instruction", "text"], ""))
    if not text:
        text = lmms_eval_specific_kwargs.get("prompt", "")
    return f"{pre_prompt}{text}{post_prompt}"


def activitynet_captions_doc_to_target(doc):
    refs = _first_present(doc, ["sentences", "caption", "captions"])
    if refs is None:
        return []
    if isinstance(refs, list):
        return [str(r) for r in refs]
    return [str(refs)]


def activitynet_captions_process_results(doc, results):
    assert len(results) == 1, f"Expected 1 result, got {len(results)}"
    pred = results[0]
    if isinstance(pred, list):
        pred = pred[0] if pred else ""
    pred = str(pred)
    refs = activitynet_captions_doc_to_target(doc)
    caption_score = {"prediction": pred, "references": refs}
    return {
        "bleu4": {"caption_score": caption_score},
        "meteor": {"caption_score": caption_score},
        "rouge_l": {"caption_score": caption_score},
        "cider": {"caption_score": caption_score},
    }


def _caption_metric_aggregate(results, metric: str) -> float:
    eval_items = [r["caption_score"] for r in results if isinstance(r, dict) and "caption_score" in r]
    if not eval_items:
        return 0.0
    if _HAS_COCO_EVAL:
        try:
            return _coco_caption_score(eval_items, metric)
        except Exception as e:
            eval_logger.warning(f"COCO eval failed: {e}. Using fallback.")
    if metric.startswith("Bleu"):
        return float(caps_bleu_aggregate(eval_items).get("bleu", 0.0))
    eval_logger.warning(f"COCO eval not available for metric {metric}.")
    return 0.0


def activitynet_captions_aggregate_bleu4(results):
    return _caption_metric_aggregate(results, "Bleu_4")


def activitynet_captions_aggregate_meteor(results):
    return _caption_metric_aggregate(results, "METEOR")


def activitynet_captions_aggregate_rouge_l(results):
    return _caption_metric_aggregate(results, "ROUGE_L")


def activitynet_captions_aggregate_cider(results):
    return _caption_metric_aggregate(results, "CIDEr")


def _extract_json_payload(text: str) -> str:
    if not text:
        return ""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(json|python)?", "", cleaned).strip()
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        return cleaned[start:end + 1]
    return cleaned


def _load_json_payload(text: str):
    payload = _extract_json_payload(text)
    if not payload:
        return None
    try:
        return json.loads(payload)
    except Exception:
        try:
            return ast.literal_eval(payload)
        except Exception:
            return None


def _chat_completion_content(response: Any) -> str:
    if isinstance(response, str):
        return response.strip()

    choices = getattr(response, "choices", None)
    if isinstance(choices, list) and choices:
        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", "")
        if content is None:
            return ""
        if isinstance(content, list):
            text_chunks = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                else:
                    text = getattr(item, "text", None)
                if isinstance(text, str):
                    text_chunks.append(text)
            return "".join(text_chunks).strip()
        return str(content).strip()

    return str(response).strip()


# -------------------- DREAM-1K (captioning) --------------------
with open(Path(__file__).parent / "dream1k.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for line in raw_data:
        if "!function" not in line:
            safe_data.append(line)
dream1k_config = yaml.safe_load("".join(safe_data))
dream1k_dataset_dir = dream1k_config["dataset_path"]


def _dream1k_get_eval_model(lmms_eval_specific_kwargs=None) -> str:
    return _get_gpt_eval_model(dream1k_config)


def dream1k_doc_to_visual(doc):
    video = doc["video_file"]
    if not video:
        return []
    candidates = [
        video,
        os.path.join(dream1k_dataset_dir, video),
    ]
    for cand in candidates:
        if cand and os.path.exists(cand):
            return [cand]
    if isinstance(video, str) and video.startswith("s3://"):
        return [video]
    sys.exit(f"video path:{video} does not exist, please check")


def dream1k_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    text = str(_first_present(doc, ["prompt", "instruction", "text"], ""))
    if not text:
        text = lmms_eval_specific_kwargs.get("prompt", "")
    return f"{pre_prompt}{text}{post_prompt}"


def dream1k_doc_to_target(doc):
    return str(doc["events"])


def _dream1k_extract_events(caption: str, model: str, retries: int = 5) -> List[str]:
    prompt = (
        "Below is a description of a video clip:\n"
        f"Video Description: {caption}\n\n"
        "Extract at most 10 key events from the above description.\n"
        "- Each event must describe an action, motion, or movement.\n"
        "- Each event should be a brief sentence within 10 words.\n"
        "- Events must be atomic and should not be repeated.\n"
        "- Scene cuts and camera motions are not events.\n"
        "- Replace pronouns with the nouns they refer to.\n\n"
        "Output JSON in the form: {\"events\": [\"event1\", \"event2\", ...]}\n"
        "Output:"
    )
    last_err = None
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=512,
            )
            content = _chat_completion_content(response)
            data = _load_json_payload(content)
            if isinstance(data, dict) and isinstance(data.get("events"), list):
                events = [str(e).strip() for e in data["events"] if str(e).strip()]
                return events
        except Exception as e:
            last_err = e
        if attempt < retries - 1:
            time.sleep(NUM_SECONDS_TO_SLEEP)
    if last_err is not None:
        eval_logger.warning(f"DREAM-1K event extraction failed: {last_err}")
    return []


def _dream1k_score_events(events: List[str], description: str, model: str, retries: int = 5) -> float:
    if not events:
        return 1.0
    prompt = (
        "Given a video description and a list of events, classify the relationship for each event as "
        "\"entailment\", \"neutral\", or \"contradiction\".\n"
        f"Video Description:\n{description}\n\n"
        f"Events: {events}\n\n"
        "Output JSON in the form:\n"
        "{\n"
        "  \"events\": [\n"
        "    {\"event\": \"event text\", \"relationship\": \"entailment\"}\n"
        "  ]\n"
        "}\n"
        "Output:"
    )
    last_err = None
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=512,
            )
            content = _chat_completion_content(response)
            data = _load_json_payload(content)
            if isinstance(data, dict) and isinstance(data.get("events"), list):
                matched = 0
                for item in data["events"]:
                    rel = str(item.get("relationship", "")).strip().lower()
                    if rel == "entailment":
                        matched += 1
                return matched / len(events)
        except Exception as e:
            last_err = e
        if attempt < retries - 1:
            time.sleep(NUM_SECONDS_TO_SLEEP)
    if last_err is not None:
        eval_logger.warning(f"DREAM-1K entailment scoring failed: {last_err}")
    return 0.0


def dream1k_process_results(doc, results, lmms_eval_specific_kwargs=None):
    assert len(results) == 1, f"Expected 1 result, got {len(results)}"
    pred = str(results[0]).strip()
    ref = str(_first_present(doc, ["description", "caption", "text"], "")).strip()
    gt_events = doc.get("events") if isinstance(doc, dict) else None
    if not isinstance(gt_events, list):
        gt_events = []
    model = _dream1k_get_eval_model(lmms_eval_specific_kwargs)

    recall = _dream1k_score_events(gt_events, pred, model)
    pred_events = _dream1k_extract_events(pred, model)
    precision = _dream1k_score_events(pred_events, ref, model)
    f1 = 0.0 if recall + precision == 0 else 2 * recall * precision / (recall + precision)

    return {
        "dream1k_f1": {"recall": recall, "precision": precision},
        "dream1k_recall": recall,
        "dream1k_precision": precision,
    }


def dream1k_aggregate_recall(results):
    values = [r for r in results if isinstance(r, (int, float))]
    return (sum(values) / len(values) * 100.0) if values else 0.0


def dream1k_aggregate_precision(results):
    values = [r for r in results if isinstance(r, (int, float))]
    return (sum(values) / len(values) * 100.0) if values else 0.0


def dream1k_aggregate_f1(results):
    recalls = []
    precisions = []
    for r in results:
        if isinstance(r, dict):
            recalls.append(r.get("recall", 0.0))
            precisions.append(r.get("precision", 0.0))
    if not recalls or not precisions:
        return 0.0
    avg_recall = sum(recalls) / len(recalls)
    avg_precision = sum(precisions) / len(precisions)
    f1 = 0.0 if avg_recall + avg_precision == 0 else 2 * avg_recall * avg_precision / (avg_recall + avg_precision)
    return f1 * 100.0


# -------------------- CapsBench (captioning) --------------------
with open(Path(__file__).parent / "capsbench.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for line in raw_data:
        if "!function" not in line:
            safe_data.append(line)
capsbench_config = yaml.safe_load("".join(safe_data))


def _capsbench_get_eval_model(lmms_eval_specific_kwargs=None) -> str:
    return _get_gpt_eval_model(capsbench_config)


def _capsbench_normalize_answer(ans: str) -> str:
    if ans is None:
        return "n/a"
    s = str(ans).strip().lower()
    if s in {"yes", "y", "true"}:
        return "yes"
    if s in {"no", "n", "false"}:
        return "no"
    if "not applicable" in s or s in {"n/a", "na", "n.a."}:
        return "n/a"
    return "n/a"


def _capsbench_parse_answers(text: str, num_questions: int) -> List[str]:
    data = _load_json_payload(text)
    answers = None
    if isinstance(data, dict):
        if isinstance(data.get("answers"), list):
            answers = data["answers"]
    elif isinstance(data, list):
        answers = data

    parsed = []
    if answers is not None:
        for item in answers:
            if isinstance(item, dict):
                parsed.append(_capsbench_normalize_answer(item.get("answer")))
            else:
                parsed.append(_capsbench_normalize_answer(item))
    else:
        tokens = re.findall(r"\b(yes|no|n/a|na|not applicable)\b", str(text).lower())
        parsed = [_capsbench_normalize_answer(tok) for tok in tokens]

    if len(parsed) < num_questions:
        parsed.extend(["n/a"] * (num_questions - len(parsed)))
    return parsed[:num_questions]


def _capsbench_call_judge(caption: str, questions: List[Dict[str, Any]], model: str, retries: int = 5) -> List[str]:
    question_lines = []
    for i, item in enumerate(questions):
        q = str(item.get("question", "")).strip()
        question_lines.append(f"{i + 1}. {q}")
    question_block = "\n".join(question_lines)

    prompt = (
        "You are given a candidate caption and a list of questions.\n"
        "Answer each question using only the information in the caption.\n"
        "Valid answers: yes, no, n/a.\n\n"
        f"Caption:\n{caption}\n\n"
        f"Questions:\n{question_block}\n\n"
        "Return JSON: {\"answers\": [\"yes\", \"no\", ...]} with the same order as the questions.\n"
        "Output:"
    )
    last_err = None
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=512,
            )
            content = _chat_completion_content(response)
            answers = _capsbench_parse_answers(content, len(questions))
            return answers
        except Exception as e:
            last_err = e
        if attempt < retries - 1:
            time.sleep(NUM_SECONDS_TO_SLEEP)
    if last_err is not None:
        eval_logger.warning(f"CapsBench judge failed: {last_err}")
    return ["n/a"] * len(questions)


def capsbench_doc_to_visual(doc):
    """Return image for CapsBench task."""
    img = doc["image"]
    if img is None:
        return []
    if isinstance(img, dict):
        if img.get("bytes"):
            try:
                return [Image.open(BytesIO(img["bytes"])).convert("RGB")]
            except Exception:
                return []
        if img.get("path"):
            try:
                return [Image.open(img["path"]).convert("RGB")]
            except Exception:
                return []
    if isinstance(img, (bytes, bytearray)):
        try:
            return [Image.open(BytesIO(img)).convert("RGB")]
        except Exception:
            return []
    if hasattr(img, "convert"):
        return [img.convert("RGB")]
    return [img]


def capsbench_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    """Return prompt text for CapsBench captioning task."""
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    text = str(_first_present(doc, ["prompt", "instruction", "text"], ""))
    if not text:
        text = lmms_eval_specific_kwargs.get("prompt", "")
    return f"{pre_prompt}{text}{post_prompt}"


def capsbench_process_results(doc, results, lmms_eval_specific_kwargs=None):
    """Process results for CapsBench captioning task."""
    assert len(results) == 1, f"Expected 1 result, got {len(results)}"
    pred = results[0]
    if isinstance(pred, list):
        pred = pred[0] if pred else ""
    pred = str(pred).strip()
    questions = doc.get("questions") if isinstance(doc, dict) else None
    if questions is None:
        questions = []
    if len(questions) == 1 and isinstance(questions[0], list):
        questions = questions[0]
    if not questions:
        return {"capsbench_acc": 0.0}

    model = _capsbench_get_eval_model(lmms_eval_specific_kwargs)
    pred_answers = _capsbench_call_judge(pred, questions, model)
    gt_answers = []
    for q in questions:
        if isinstance(q, dict):
            gt_answers.append(_capsbench_normalize_answer(q.get("answer")))
        else:
            gt_answers.append("n/a")
    if not gt_answers:
        score = 0.0
    else:
        correct = sum(1 for p, g in zip(pred_answers, gt_answers) if p == g)
        score = correct / len(gt_answers)

    return {"capsbench_acc": score}


def capsbench_aggregate_accuracy(results):
    values = [r for r in results if isinstance(r, (int, float))]
    return (sum(values) / len(values) * 100.0) if values else 0.0


# -------------------- Caption metrics aggregation --------------------
def _get_ngrams(sentence: str, n: int) -> Counter:
    """Get n-grams from sentence."""
    if sentence is None:
        return Counter()
    tokens = str(sentence).strip().split()
    if len(tokens) < n:
        return Counter()
    grams = zip(*(tokens[i:] for i in range(n)))
    return Counter([" ".join(g) for g in grams])


def caps_bleu_aggregate(eval_items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute BLEU score (fallback implementation)."""
    max_n = 4
    total_matches = [0] * max_n
    total_candidates = [0] * max_n
    ref_length = 0
    cand_length = 0
    n = 0
    
    for it in eval_items:
        cand = it.get("prediction")
        refs = it.get("references") or []
        if cand is None or not refs:
            continue
        
        cand_str = str(cand).strip()
        ref_strs = [str(r).strip() for r in refs]
        cand_tokens = cand_str.split()
        cand_length += len(cand_tokens)
        
        ref_lens = [len(r.split()) for r in ref_strs]
        if ref_lens:
            closest = min(ref_lens, key=lambda x: (abs(x - len(cand_tokens)), x))
            ref_length += closest
        
        n += 1
        
        for i in range(1, max_n + 1):
            cand_ngrams = _get_ngrams(cand_str, i)
            total_candidates[i - 1] += sum(cand_ngrams.values())
            
            max_ref_counts = Counter()
            for r in ref_strs:
                ref_ngrams = _get_ngrams(r, i)
                for k, v in ref_ngrams.items():
                    if v > max_ref_counts[k]:
                        max_ref_counts[k] = v
            
            for gram, cnt in cand_ngrams.items():
                total_matches[i - 1] += min(cnt, max_ref_counts.get(gram, 0))
    
    # Compute BLEU score
    precisions = []
    for i in range(max_n):
        if total_candidates[i] == 0:
            precisions.append(0.0)
        else:
            precisions.append(total_matches[i] / total_candidates[i])
    
    smooth = 1e-9
    log_prec_sum = sum(math.log(p if p > 0 else smooth) for p in precisions)
    geo_mean = math.exp(log_prec_sum / max_n)
    
    if cand_length == 0 or n == 0:
        bp = 0.0
    else:
        bp = 1.0 if cand_length > ref_length else math.exp(1 - ref_length / (cand_length + 1e-9))
    
    bleu = bp * geo_mean
    return {"bleu": bleu, "n": n}


def capsbench_calculate_BLEU(results):
    """Aggregate caption metrics for CapsBench."""
    # Extract caption score items
    eval_items = [r["caption_score"] for r in results if "caption_score" in r]
    
    if not eval_items:
        return {"error": "No valid items for aggregation"}
    
    coco_name = "Bleu_4"
    if _HAS_COCO_EVAL:
        try:
            score = _coco_caption_score(eval_items, coco_name)
            return {f"{coco_name.lower()}": score, "n": len(eval_items)}
        except Exception as e:
            eval_logger.warning(f"COCO eval failed: {e}. Using fallback.")

    return caps_bleu_aggregate(eval_items)

def capsbench_calculate_CIDEr(results):
    """Aggregate caption metrics for CapsBench."""
    eval_items = [r["caption_score"] for r in results if "caption_score" in r]
    
    if not eval_items:
        return {"error": "No valid items for aggregation"}

    if _HAS_COCO_EVAL:
        try:
            score = _coco_caption_score(eval_items, "CIDEr")
            return {"cider": score, "n": len(eval_items)}
        except Exception as e:
            eval_logger.warning(f"COCO eval failed: {e}. Using fallback.")

    return caps_bleu_aggregate(eval_items)

def capsbench_aggregate_results_BLEU(results):
    value = 0.0
    length = 0
    for r in results:
        if "BLEU_4" in r:
            value += r["BLEU_4"]
            length += 1
    return value / length if length > 0 else 0.0

def capsbench_aggregate_results_CIDEr(results):
    value = 0.0
    length = 0
    for r in results:
        if "CIDEr" in r:
            value += r["CIDEr"]
            length += 1
    return value / length if length > 0 else 0.0

def _coco_caption_score(eval_items: List[Dict[str, Any]], metric: str) -> float:
    """Helper to compute COCO caption metrics."""
    dataset = {"annotations": [], "images": []}
    stored_results = []
    
    for idx, item in enumerate(eval_items):
        pred = str(item.get("prediction", ""))
        refs = item.get("references") or []
        
        stored_results.append({"image_id": idx, "caption": pred})
        
        ann_id = len(dataset["annotations"])
        for ref in refs:
            dataset["annotations"].append({
                "image_id": idx,
                "caption": str(ref),
                "id": ann_id
            })
            ann_id += 1
        
        dataset["images"].append({"id": idx})
    
    coco = COCO()
    coco.dataset = dataset
    coco.createIndex()
    
    coco_res = coco.loadRes(stored_results)
    coco_eval = COCOEvalCap(coco, coco_res)
    
    imgIds = coco_eval.params["image_id"]
    gts = {imgId: coco_eval.coco.imgToAnns[imgId] for imgId in imgIds}
    res = {imgId: coco_eval.cocoRes.imgToAnns[imgId] for imgId in imgIds}
    
    tokenizer = PTBTokenizer()
    gts = tokenizer.tokenize(gts)
    res = tokenizer.tokenize(res)
    
    if metric.startswith("Bleu"):
        scorer = Bleu(4)
        score, _ = scorer.compute_score(gts, res)
        n = int(metric.split("_")[-1])
        return float(score[n - 1])
    elif metric == "CIDEr":
        scorer = Cider()
        score, _ = scorer.compute_score(gts, res)
        return float(score)
    elif metric == "METEOR":
        scorer = Meteor()
        score, _ = scorer.compute_score(gts, res)
        return float(score)
    elif metric == "ROUGE_L":
        scorer = Rouge()
        score, _ = scorer.compute_score(gts, res)
        return float(score)
    else:
        raise ValueError(f"Unsupported metric: {metric}")


# -------------------- Generic aggregators --------------------
def accuracy_aggregate_results(results):
    """Compute accuracy from exact_match scores."""
    scores = [r.get("exact_match", 0.0) for r in results if "exact_match" in r]
    if not scores:
        return 0.0
    return statistics.mean(scores)


# -------------------- Video MMMU --------------------
with open(Path(__file__).parent / "videommmu_adaptation.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)

    videommmu_config = yaml.safe_load("".join(safe_data))
videommmu_dataset_dir = videommmu_config["dataset_path"]

# Define the mapping for subjects to their respective directories
def get_cache_dir_videommmu(subject):
    if subject in ["Art", "Art_Theory", "Design", "Music"]:
        return "Art"
    elif subject in ["Biology", "Chemistry", "Geography", "Math", "Physics"]:
        return "Science"
    elif subject in ["History", "Literature", "Sociology", "Psychology"]:
        return "Humanities"
    elif subject in ["Agriculture", "Architecture_and_Engineering", "Computer_Science", "Electronics", "Energy_and_Power", "Materials", "Mechanical_Engineering"]:
        return "Engineering"
    elif subject in ["Basic_Medical_Science", "Clinical_Medicine", "Diagnostics_and_Laboratory_Medicine", "Pharmacy", "Public_Health"]:
        return "Medicine"
    elif subject in ["Accounting", "Economics", "Finance", "Manage", "Marketing"]:
        return "Business"
    else:
        raise ValueError(f"Subject {subject} not recognized.")


def videommmu_doc_to_visual(doc):
    # Extract the subject between the first and last underscores
    subject = "_".join(doc["id"].split("_")[1:-1])

    # Get the appropriate cache directory based on the subject
    videommmu_cache_dir = os.path.join(videommmu_dataset_dir, get_cache_dir_videommmu(subject))

    video_path = doc["id"] + ".mp4"
    video_path = os.path.join(videommmu_cache_dir, video_path)

    if os.path.exists(video_path):
        video_path = video_path
    else:
        sys.exit(f"video path:{video_path} does not exist, please check")

    return [video_path]


def videommmu_doc_to_visual_question_only(doc):
    video_path = doc["id"] + "_image" + ".mp4"
    question_only_cache_dir = os.path.join(videommmu_dataset_dir, "question_only")
    video_path = os.path.join(question_only_cache_dir, video_path)

    if os.path.exists(video_path):
        video_path = video_path
    else:
        sys.exit(f"video path:{video_path} does not exist, please check")

    return [video_path]


# This is the place to format the input
def videommmu_doc_to_text_adaptation(doc, lmms_eval_specific_kwargs=None):
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    pre_prompt = ""
    post_prompt = ""

    pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    question = doc["question"]

    if doc["question_type"] == "multiple-choice":
        pre_prompt += lmms_eval_specific_kwargs["mcq_prompt"]
        parsed_options = parse_options_videommmu(doc["options"])
        question += "\n" + parsed_options
    else:
        pre_prompt += lmms_eval_specific_kwargs["open_ended_prompt"]

    return f"{pre_prompt}{question}"


def videommmu_doc_to_text_no_preprompt(doc, lmms_eval_specific_kwargs=None):
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}

    question = doc["question"]
    parsed_options = parse_options_videommmu(doc["options"])
    question += "\n" + parsed_options

    return f"{question}"


def videommmu_doc_to_text_perception_comprehension(doc, lmms_eval_specific_kwargs=None):
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    post_prompt = ""
    post_prompt += lmms_eval_specific_kwargs["perception_and_comprehension_prompt"]
    question = doc["question"]
    parsed_options = parse_options_videommmu(doc["options"])
    question += "\n" + parsed_options

    return f"{question}{post_prompt}"


def parse_options_videommmu(options):
    # Define the option letters based on the number of options
    option_letters = [chr(ord("A") + i) for i in range(len(options))]

    # Check if the options are already appended with letters
    if all(option.startswith(f"{letter}.") for option, letter in zip(options, option_letters)):
        return "\n".join(options)

    # Otherwise, append option letters
    choices_str = "\n".join([f"{option_letter}. {option}" for option_letter, option in zip(option_letters, options)])
    return choices_str


def videommmu_doc_to_answer(doc):
    return doc["answer"]


# process results for each individual instance
def videommmu_process_results(doc, results):
    pred = results[0]

    question_type = doc.get("question_type", "None")
    if question_type == "multiple-choice":
        index2ans, all_choices = get_multi_choice_info_videommmu((doc["options"]))
        parsed_pred = parse_multi_choice_response_videommmu(pred, all_choices, index2ans)
    else:
        parsed_pred = parse_open_response_videommmu(pred)

    mmmu_acc = {"id": doc["id"], "subdomain": extract_subset_name(doc["id"]), "question_type": question_type, "answer": doc["answer"], "parsed_pred": parsed_pred}
    return {"mmmu_acc": mmmu_acc}


def videommmu_aggregate_results(results):
    evaluation_result = {}
    subset_to_eval_samples = defaultdict(list)

    # Filter out results where parsed_pred is "API Error"
    valid_results = [result for result in results if result["parsed_pred"] != "API Error"]

    for result in valid_results:
        subset_to_eval_samples[result["subdomain"]].append(result)

    for subset, sub_eval_samples in subset_to_eval_samples.items():
        judge_dict, metric_dict = evaluate_videommmu(sub_eval_samples)
        metric_dict.update({"num_example": len(sub_eval_samples)})
        evaluation_result[subset] = metric_dict

    printable_results = {}
    for domain, in_domain_cats in DOMAIN_CAT2SUB_CAT.items():
        in_domain_cat_results = {}
        for cat_name in in_domain_cats:
            if cat_name in evaluation_result.keys():
                in_domain_cat_results[cat_name] = evaluation_result[cat_name]
        in_domain_ins_acc = calculate_ins_level_acc(in_domain_cat_results)
        in_domain_data_num = sum([cat_results["num_example"] for cat_results in in_domain_cat_results.values()])
        printable_results["Overall-" + domain] = {
            "num": int(in_domain_data_num),
            "acc": round(in_domain_ins_acc, 5),
        }
        # Add subcategory
        for cat_name, cat_results in in_domain_cat_results.items():
            printable_results[cat_name] = {
                "num": int(cat_results["num_example"]),
                "acc": round(cat_results["acc"], 5),
            }
    all_ins_acc = calculate_ins_level_acc(evaluation_result)
    printable_results["Overall"] = {
        "num": sum([cat_results["num_example"] for cat_results in evaluation_result.values()]),
        "acc": round(all_ins_acc, 5),
    }
    print(printable_results)
    return printable_results["Overall"]["acc"]


##################
# Helper functions.
##################

DOMAIN_CAT2SUB_CAT = {
    "Art and Design": ["Art", "Art_Theory", "Design", "Music"],
    "Business": ["Accounting", "Economics", "Finance", "Manage", "Marketing"],
    "Science": [
        "Biology",
        "Chemistry",
        "Geography",
        "Math",
        "Physics",
    ],
    "Health and Medicine": [
        "Basic_Medical_Science",
        "Clinical_Medicine",
        "Diagnostics_and_Laboratory_Medicine",
        "Pharmacy",
        "Public_Health",
    ],
    "Humanities and Social Science": [
        "History",
        "Literature",
        "Sociology",
        "Psychology",
    ],
    "Tech and Engineering": [
        "Agriculture",
        "Architecture_and_Engineering",
        "Computer_Science",
        "Electronics",
        "Energy_and_Power",
        "Materials",
        "Mechanical_Engineering",
    ],
}

# Batch evaluation for multiple choice and open questions.
def evaluate_videommmu(samples):
    pred_correct = 0
    judge_dict = dict()
    for sample in samples:
        gold_i = sample["answer"]
        pred_i = sample["parsed_pred"]
        if sample["question_type"] == "multiple-choice":
            correct = eval_multi_choice(gold_i, pred_i)
        elif sample["question_type"] == "perception":
            correct = eval_multi_choice(gold_i, pred_i)
        else:  # open question
            correct = eval_open(gold_i, pred_i)

        if correct:
            judge_dict[sample["id"]] = "Correct"
            pred_correct += 1
        else:
            judge_dict[sample["id"]] = "Wrong"

    if len(samples) == 0:
        return {"acc": 0}
    return judge_dict, {"acc": pred_correct / len(samples)}


def parse_multi_choice_response_videommmu(response, all_choices, index2ans):
    """
    Parse the prediction from the generated response.
    Return the predicted index e.g., A, B, C, D.
    """
    if response == "API Error" or response == "":
        return "API Error"

    # Step 1: Clean up punctuation from the response
    for char in [",", ".", "!", "?", ";", ":", "'"]:
        response = response.strip(char)
    response = " " + response + " "  # Add space to avoid partial match
    # print(response)

    index_ans = True
    ans_with_brack = False
    ans_with_period = False
    ans_with_colon = False
    candidates = []

    # Step 2: If no candidates, look for choices with a period after (A. B. C. D.)
    for choice in all_choices:  # e.g., A. B. C. D.
        if f"{choice}." in response:
            candidates.append(choice)
            ans_with_period = True
    # Step 2.1: If no candidates, look for choices with a colon after (A: B: C: D:)
    for choice in all_choices:  # e.g., A: B: C: D:
        if f"{choice}:" in response:
            candidates.append(choice)
            ans_with_colon = True
    # Step 3: Look for choices with parentheses e.g., (A) (B) (C) (D)
    if len(candidates) == 0:
        for choice in all_choices:  # e.g., (A) (B) (C) (D)
            if f"({choice})" in response:
                candidates.append(choice)
                ans_with_brack = True
    # Step 4: If no candidates, look for choices with a space after (A B C D)
    if len(candidates) == 0:
        for choice in all_choices:  # e.g., A B C D
            if f"{choice} " in response:
                candidates.append(choice)

    # Step 5: If no candidates and response has more than 5 tokens, try parsing based on content
    if len(candidates) == 0 and len(response.split()) > 5:
        for index, ans in index2ans.items():
            if ans.lower() in response.lower():
                candidates.append(index)
                index_ans = False  # It's content answer, not an index

    # Step 6: If still no candidates, randomly choose one
    if len(candidates) == 0:
        pred_index = "No Answer Found."

    # Step 7: If multiple candidates found, use the one appearing last
    elif len(candidates) > 1:
        start_indexes = []
        if index_ans:
            if ans_with_period:
                for can in candidates:
                    index = response.rfind(f"{can}.")
                    start_indexes.append(index)
            elif ans_with_colon:
                for can in candidates:
                    index = response.rfind(f"{can}:")
                    start_indexes.append(index)
            elif ans_with_brack:
                for can in candidates:
                    index = response.rfind(f"({can})")
                    start_indexes.append(index)
            else:
                for can in candidates:
                    index = response.rfind(f" {can} ")
                    start_indexes.append(index)
        else:
            for can in candidates:
                index = response.lower().rfind(index2ans[can].lower())
                start_indexes.append(index)
        # Get the last one (max index)
        pred_index = candidates[np.argmax(start_indexes)]
    else:
        # If only one candidate, use it
        pred_index = candidates[0]

    return pred_index

def parse_open_response_videommmu(response):
    """
    Parse the prediction from the generated response.
    Return a list of predicted strings or numbers.
    """
    if response == "API Error" or response == "":
        return "API Error"

    # content = content.strip("\n").strip(".").strip(" ")
    def get_key_subresponses(response):
        key_responses = []
        response = response.strip().strip(".").lower()
        sub_responses = re.split(r"\.\s(?=[A-Z])|\n", response)
        indicators_of_keys = [
            # Common explanation or conclusion phrases
            "could be ",
            "so ",
            "is ",
            "thus ",
            "therefore ",
            "final ",
            "answer ",
            "result ",
            "are ",
            "in total ",
            "total ",
            "identify ",
            "recognize ",
            "calculated as ",
            "counted as ",
            "measured as ",
            "observed as ",
            "concluded as ",
            "found to be ",
            "equals ",
            "determined to be ",
            "number of ",
            "value is ",
            "adds up to ",
            "have ",
            "has ",
        ]

        key_responses = []
        for index, resp in enumerate(sub_responses):
            # if last one, accept it's an equation (the entire response can be just one sentence with equation)
            if index == len(sub_responses) - 1:
                indicators_of_keys.extend(["="])
            shortest_key_response = None  # the shortest response that may contain the answer (tail part of the response)
            for indicator in indicators_of_keys:
                if indicator in resp:
                    if not shortest_key_response:
                        shortest_key_response = resp.split(indicator)[-1].strip()
                    else:
                        if len(resp.split(indicator)[-1].strip()) < len(shortest_key_response):
                            shortest_key_response = resp.split(indicator)[-1].strip()
                    # key_responses.append(resp.split(indicator)[1].strip())

            if shortest_key_response:
                # and it's not trivial
                if shortest_key_response.strip() not in [
                    ":",
                    ",",
                    ".",
                    "!",
                    "?",
                    ";",
                    ":",
                    "'",
                ]:
                    key_responses.append(shortest_key_response)
        if len(key_responses) == 0:  # did not found any
            return [response]
        return key_responses

    key_responses = get_key_subresponses(response)

    pred_list = key_responses.copy()  # keep the original string response
    for resp in key_responses:
        pred_list.extend(extract_numbers(resp))

    tmp_pred_list = []
    for i in range(len(pred_list)):
        tmp_pred_list.extend(normalize_str(pred_list[i]))
    pred_list = tmp_pred_list

    # remove duplicates
    pred_list = list(set(pred_list))

    return pred_list


def get_multi_choice_info_videommmu(options):
    """
    Given the list of options for multiple choice question
    Return the index2ans and all_choices
    """

    start_chr = "A"
    all_choices = []
    index2ans = {}
    for i, option in enumerate(options):
        index2ans[chr(ord(start_chr) + i)] = option
        all_choices.append(chr(ord(start_chr) + i))

    return index2ans, all_choices