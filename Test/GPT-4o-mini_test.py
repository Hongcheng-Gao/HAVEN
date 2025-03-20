
import pandas as pd
import argparse
import os, sys
sys.path.append(os.getcwd())
from tqdm import tqdm
import numpy as np
from transformers import set_seed
import torch
from openai import OpenAI
import os
import time
import cv2
import requests
import base64
def gen_ans(args):
    api_key = args.api_key
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    question_data = pd.read_csv(args.data_path,encoding='ISO-8859-1')
    question_data[f"pred_ans"] = None
    base_video_path = args.video_path
    start = args.start
    end = args.end     
    choice_instruction = '''The following is a multiple-choice question. Please choose one of the following options. '''
    judge_instruction = '''Please answer the following yes/no question. '''
    dataset = args.dataset
    output_file = args.output_file
    if not os.path.exists(output_file):
        os.makedirs(output_file)
    start_idx = args.start
    end_idx = args.end
    save_file = f'{output_file}/GPT4o-mini_test.csv'
    for i in tqdm(range(start_idx, end_idx)):
        question = question_data.iloc[i]['Question']
        video_id = question_data.iloc[i]['video_id']
        video_path = os.path.join(base_video_path, f"{video_id}.mp4")
        # print('video_path',video_path)
        if question_data.iloc[i]['Form'] == 'choice':
            instruction = choice_instruction + "\n" + question
        elif question_data.iloc[i]['Form'] == 'tf':
            instruction = judge_instruction + "\n" + question
        else:
            instruction = question

        video = cv2.VideoCapture(video_path)
        base64Frames = []
        while video.isOpened():
            success, frame = video.read()
            if not success:
                break
            _, buffer = cv2.imencode(".jpg", frame)
            base64Frames.append(base64.b64encode(buffer).decode("utf-8"))
        video.release()
        print(len(base64Frames), "frames read.")
        num_samples = args.num_samples
        total_frames = len(base64Frames)
        sample_indices = np.linspace(0, total_frames - 1, num_samples, dtype=int)
        sampled_frames = [base64Frames[i] for i in sample_indices]
        PROMPT_MESSAGES = [
            {
                "role": "user",
                "content": [
                    instruction,
                    *map(lambda x: {"image": x, "resize": 768}, sampled_frames),
                ],
            },
        ]
        payload_2 = {
                "model": "gpt-4o-mini",
                "messages": PROMPT_MESSAGES,
                "max_tokens": 200
            }
        res_ori = requests.post(args.GPT_url, headers=headers, json=payload_2)
        correct_pro = res_ori.json()
        pred = correct_pro["choices"][0]["message"]["content"].strip()
        print(f'Instruction:\t{instruction}')
        print(f'Answer:\t{pred}')
        print('-'*20)
        question_data.loc[i, f"pred_ans"] = pred
        question_data.to_csv(save_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument("--start", type=int)
    parser.add_argument("--end", type=int)
    parser.add_argument("--output_file", type=str, default="")
    parser.add_argument("--api_key", type=str, default="")
    parser.add_argument("--GPT_url", type=str, default="")
    args = parser.parse_args()
    set_seed(args.seed)
    gen_ans(args)






