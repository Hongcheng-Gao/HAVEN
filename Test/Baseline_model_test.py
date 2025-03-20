import os, sys
import json
import argparse
sys.path.append(os.getcwd())
from tqdm import tqdm
import pandas as pd
from transformers import set_seed

def eval_model(args):
    configs = json.load(open("HAVEN/baselines/config.json"))
    CKPT_DIR = configs['CKPT_DIR']
    TESTING_MODEL=args.model_name
    def load_model(TESTING_MODEL):
        if TESTING_MODEL == 'VideoChatGPT':
            from HAVEN.baselines.videochatgpt_modeling import VideoChatGPT
            ckpt_path = f"{CKPT_DIR}/Video-ChatGPT-7B"
            model = VideoChatGPT({"model_path": ckpt_path, "device": 0})
        elif TESTING_MODEL == "Valley2":
            from HAVEN.baselines.valley_modeling import Valley
            ckpt_path = f"{CKPT_DIR}/Valley2-7b"
            model = Valley({"model_path": ckpt_path, "device": 0})
        elif TESTING_MODEL == "Video-LLaMA-2-7B":
            from HAVEN.baselines.videollama_modeling import VideoLLaMA
            ckpt_path = f"{CKPT_DIR}/Video-LLaMA-2-7B-Finetuned"
            model = VideoLLaMA({"model_path": ckpt_path, "device": 0})
        elif TESTING_MODEL == "Video-LLaMA-2-13B":
            from HAVEN.baselines.videollama_modeling import VideoLLaMA
            ckpt_path = f"{CKPT_DIR}/Video-LLaMA-2-13B-Finetuned"
            model = VideoLLaMA({"model_path": ckpt_path, "device": 0})
        elif TESTING_MODEL == "VideoChat2":
            from HAVEN.baselines.videochat_modeling import VideoChat
            ckpt_path = f"{CKPT_DIR}/VideoChat2"
            model = VideoChat({"model_path": ckpt_path, "device": 0})
        elif TESTING_MODEL == "LLaMA-VID":
            from HAVEN.baselines.llamavid_modeling import LLaMAVID
            ckpt_path = f"{CKPT_DIR}/LLaMA-VID-7B"
            model = LLaMAVID({"model_path": ckpt_path, "device": 0})
        elif TESTING_MODEL == "LLaMA-VID-13B":
            from HAVEN.baselines.llamavid_modeling import LLaMAVID
            ckpt_path = f"{CKPT_DIR}/LLaMA-VID-13B"
            model = LLaMAVID({"model_path": ckpt_path, "device": 0})
        elif TESTING_MODEL == "PLLaVA":
            from HAVEN.baselines.pllava_modeling import PLLaVA
            ckpt_path = f"{CKPT_DIR}/PLLaVA/pllava-7b"
            model = PLLaVA({"model_path": ckpt_path, "device": 0})
        elif TESTING_MODEL == "PLLaVA-13B":
            from HAVEN.baselines.pllava_modeling import PLLaVA
            ckpt_path = f"{CKPT_DIR}/PLLaVA/pllava-13b"
            model = PLLaVA({"model_path": ckpt_path, "device": 0})
        elif TESTING_MODEL == "LLaVA-NeXT-Video":
            from HAVEN.baselines.llavanext_modeling import LLaVANeXT
            ckpt_path = "llava-hf/LLaVA-NeXT-Video-7B-DPO-hf"
            model = LLaVANeXT({"model_path": ckpt_path, "device": 2})
        elif TESTING_MODEL == "LLaVA-NeXT-Video-34B":
            from HAVEN.baselines.llavanext_modeling import LLaVANeXT
            ckpt_path = f"{CKPT_DIR}/LLaVA-NeXT-Video/LLaVA-NeXT-Video-34B-DPO"
            model = LLaVANeXT({"model_path": ckpt_path, "device": 0})
        elif TESTING_MODEL == "ShareGPT4Video":
            from HAVEN.baselines.sharegpt4video_modeling import ShareGPT4Video
            ckpt_path = f"{CKPT_DIR}/ShareGPT4Video/sharegpt4video-8b"
            model = ShareGPT4Video({"model_path": ckpt_path, "device": 0})
        elif TESTING_MODEL == "LLaVA":
            from HAVEN.baselines.llava_modeling import LLaVA
            ckpt_path = f"{CKPT_DIR}/LLaVA/llava-v1.5-7b"
            model = LLaVA({"model_path": ckpt_path, "device": 0})
        return model

    dataset = args.dataset
    question_data = pd.read_csv(args.dataset_path,encoding='ISO-8859-1')
    question_data[f"pred_ans"] = None
    base_video_path = args.video_path
    choice_instruction = '''The following is a multiple-choice question. Please choose one of the following options. '''
    judge_instruction = '''Please answer the following yes/no question. '''
    model = load_model(TESTING_MODEL)

    start = args.start
    end = args.end
    output_file = args.output_file
    save_file = f'{output_file}/COT_{args.model_name}_{dataset}_{start}.csv'
    for i in tqdm(range(start,end)):
        question = question_data.iloc[i]['Question']
        video_id = question_data.iloc[i]['video_id']
        video_path = os.path.join(base_video_path, f"{video_id}.mp4")
        if question_data.iloc[i]['Form'] == 'choice':
            instruction = choice_instruction + "\n" + question
        elif question_data.iloc[i]['Form'] == 'tf':
            instruction = judge_instruction + "\n" + question
        else:
            instruction = question

        pred = model.generate(
                    instruction=instruction,
                    video_path=video_path,
                )
        print(f'Instruction:\t{instruction}')
        print(f'Answer:\t{pred}')
        print('-'*20)
        question_data.loc[i, f"pred_ans"] = pred
        question_data.to_csv(save_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_name", type=str,default="LLaVA-NeXT-Video")
    parser.add_argument("--dataset_path", type=str,default='')
    parser.add_argument("--video_path", type=str,default='')
    parser.add_argument("--start", type=int, default="0")
    parser.add_argument("--end", type=int, default="-1")
    parser.add_argument("--output_file", type=str, default="")
    args = parser.parse_args()
    set_seed(args.seed)
    eval_model(args)