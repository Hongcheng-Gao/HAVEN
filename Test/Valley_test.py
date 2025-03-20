from HAVEN.baselines.valley_eagle_chat import ValleyEagleChat
import decord
from torchvision import transforms
import pandas as pd
import argparse
import os, sys
sys.path.append(os.getcwd())
from tqdm import tqdm
import numpy as np
from transformers import set_seed
import torch
def eval_model(args):
    model = ValleyEagleChat(
        model_path='bytedance-research/Valley-Eagle-7B',
        torch_dtype=torch.bfloat16,
        padding_side = 'left',
    )
    question_data = pd.read_csv(args.data_path,encoding='ISO-8859-1')
    question_data[f"pred_ans"] = None
    base_video_path = args.video_path
    start = args.start
    end = args.end     
    choice_instruction = '''The following is a multiple-choice question. Please choose one of the following options. '''
    judge_instruction = '''Please answer the following yes/no question. '''
    output_file = args.output_file
    if not os.path.exists(output_file):
        os.makedirs(output_file)
    num_frame = args.num_frame
    start_idx = args.start
    end_idx = args.end
    save_file = f'{output_file}/Valley_test.csv'
    for i in tqdm(range(start_idx, end_idx)):
        question = question_data.iloc[i]['Question']
        video_id = question_data.iloc[i]['video_id']
        video_path = os.path.join(base_video_path, f"{video_id}.mp4")
        if question_data.iloc[i]['Form'] == 'choice':
            instruction = choice_instruction + "\n" + question
        elif question_data.iloc[i]['Form'] == 'tf':
            instruction = judge_instruction + "\n" + question
        else:
            instruction = question
        video_reader = decord.VideoReader(video_path)
        decord.bridge.set_bridge("torch")
        video = video_reader.get_batch(
            np.linspace(0,  len(video_reader) - 1, num_frame).astype(np.int_)
        ).byte()
        request = {
            "chat_history": [
                {'role': 'system', 'content': 'You are Valley, developed by ByteDance. Your are a helpfull Assistant.'},
                {'role': 'user', 'content': instruction},
            ],
            "images": [transforms.ToPILImage()(image.permute(2, 0, 1)).convert("RGB") for image in video],
        }
        pred = model(request)
        print(f"\n>>> Assistant:\n")
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