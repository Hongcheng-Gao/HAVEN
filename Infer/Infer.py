import os, sys
import json
import argparse
sys.path.append(os.getcwd())
from tqdm import tqdm
import pandas as pd
from transformers import set_seed

def load_model(TESTING_MODEL):
    configs = json.load(open("baselines/config.json"))
    CKPT_DIR = configs['CKPT_DIR']
    TESTING_MODEL = args.model_name
    if TESTING_MODEL == 'VideoChatGPT':
        from baselines.videochatgpt_modeling import VideoChatGPT
        ckpt_path = f"{CKPT_DIR}/Video-ChatGPT-7B"
        model = VideoChatGPT({"model_path": ckpt_path, "device": 0})
    elif TESTING_MODEL == "Video-LLaMA-2-7B":
        from baselines.videollama_modeling import VideoLLaMA
        ckpt_path = f"{CKPT_DIR}/Video-LLaMA-2-7B-Finetuned"
        model = VideoLLaMA({"model_path": ckpt_path, "device": 0})
    elif TESTING_MODEL == "Video-LLaMA-2-13B":
        from baselines.videollama_modeling import VideoLLaMA
        ckpt_path = f"{CKPT_DIR}/Video-LLaMA-2-13B-Finetuned"
        model = VideoLLaMA({"model_path": ckpt_path, "device": 0})
    elif TESTING_MODEL == "VideoChat2":
        from baselines.videochat_modeling import VideoChat
        ckpt_path = f"{CKPT_DIR}/VideoChat2"
        model = VideoChat({"model_path": ckpt_path, "device": 0})
    elif TESTING_MODEL == "LLaMA-VID":
        from baselines.llamavid_modeling import LLaMAVID
        ckpt_path = f"{CKPT_DIR}/LLaMA-VID-7B"
        model = LLaMAVID({"model_path": ckpt_path, "device": 0})
    elif TESTING_MODEL == "LLaMA-VID-13B":
        from baselines.llamavid_modeling import LLaMAVID
        ckpt_path = f"{CKPT_DIR}/LLaMA-VID-13B"
        model = LLaMAVID({"model_path": ckpt_path, "device": 0})
    elif TESTING_MODEL == "PLLaVA":
        from baselines.pllava_modeling import PLLaVA
        ckpt_path = f"{CKPT_DIR}/PLLaVA/pllava-7b"
        model = PLLaVA({"model_path": ckpt_path, "device": 0})
    elif TESTING_MODEL == "PLLaVA-13B":
        from baselines.pllava_modeling import PLLaVA
        ckpt_path = f"{CKPT_DIR}/PLLaVA/pllava-13b"
        model = PLLaVA({"model_path": ckpt_path, "device": 0})
    elif TESTING_MODEL == "LLaVA-NeXT-Video":
        from baselines.llavanext_modeling import LLaVANeXT
        ckpt_path = "llava-hf/LLaVA-NeXT-Video-7B-DPO-hf"
        model = LLaVANeXT({"model_path": ckpt_path, "device": 2})
    elif TESTING_MODEL == "LLaVA-NeXT-Video-34B":
        from baselines.llavanext_modeling import LLaVANeXT
        ckpt_path = f"{CKPT_DIR}/LLaVA-NeXT-Video/LLaVA-NeXT-Video-34B-DPO"
        model = LLaVANeXT({"model_path": ckpt_path, "device": 0})
    elif TESTING_MODEL == "ShareGPT4Video":
        from baselines.sharegpt4video_modeling import ShareGPT4Video
        ckpt_path = f"{CKPT_DIR}/ShareGPT4Video/sharegpt4video-8b"
        model = ShareGPT4Video({"model_path": ckpt_path, "device": 0})
    elif TESTING_MODEL == "LLaVA":
        from baselines.llava_modeling import LLaVA
        ckpt_path = f"{CKPT_DIR}/LLaVA/llava-v1.5-7b"
        model = LLaVA({"model_path": ckpt_path, "device": 0})
    elif TESTING_MODEL == "Qwen2.5-VL":
        from baselines.qwen_modeling import Qwen_VL
        ckpt_path = f"{CKPT_DIR}/Qwen/Qwen2.5-VL-7B-Instruct"
        model = Qwen_VL(model_path=ckpt_path, device="cuda")
    elif TESTING_MODEL == "Valley":
        from baselines.valley_modeling import Valley
        ckpt_path = f"{CKPT_DIR}/bytedance-research/Valley-Eagle-7B"
        model = Valley(model_path=ckpt_path, device="cuda")
    return model

def eval_model(args):
    TESTING_MODEL = args.model_name
    with open(args.dataset_path, "r", encoding="utf-8") as f:
        question_data = json.load(f)
    base_video_path = args.video_path
    choice_instruction = '''The following is a multiple-choice question. Please choose one of the following options. '''
    judge_instruction = '''Please answer the following yes/no question. '''
    model = load_model(TESTING_MODEL)
    output_file = args.output_file
    save_file = f'{output_file}/{args.model_name}_test_result.json'
    for i in tqdm(range(len(question_data))):
        question = question_data[i]['Question']
        video_id = question_data[i]['Video Path']
        video_path = os.path.join(base_video_path, f"{video_id}.mp4")
        if question_data[i]['Form'] == 'Multiple-choice':
            instruction = choice_instruction + "\n" + question
        elif question_data[i]['Form'] == 'Binary-choice':
            instruction = judge_instruction + "\n" + question
        else:
            instruction = question
        pred = model.generate(
                    instruction=instruction,
                    video_path=video_path,
                )
        question_data[i]["Model Answer"] = pred
    with open(save_file, "w", encoding="utf-8") as f:
        json.dump(question_data, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_name", type=str,default="LLaVA-NeXT-Video",
                        choices=["VideoChatGPT", "Video-LLaMA-2", "VideoChat2", "VideoLLaVA", "LLaMA-VID",
                                 "PLLaVA", "LLaVA-NeXT-Video", "ShareGPT4Video", "LLaVA", "Qwen2.5-VL",
                                 "Video-LLaMA-2-7B","Video-LLaMA-2-13B", "LLaMA-VID-13B", "PLLaVA-13B",
                                 "LLaVA-NeXT-Video-34B", "Valley"]
                        )
    parser.add_argument("--dataset_path", type=str,default='Data/test_data.json')
    parser.add_argument("--video_path", type=str, help="path to downloaded video file")
    parser.add_argument("--output_file", type=str, help="path to output file")
    args = parser.parse_args()
    set_seed(args.seed)
    eval_model(args)