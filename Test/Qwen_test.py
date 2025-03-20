
import pandas as pd
import argparse
import os, sys
sys.path.append(os.getcwd())
from tqdm import tqdm
import numpy as np
from transformers import set_seed
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

def gen_ans(args):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    # default processer
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
    question_data = args.data_path
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
    save_file = f'{output_file}/Qwen2.5_test.csv'
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
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                    },
                    {"type": "text", "text": instruction},
                ],
            }
        ]
        # Preparation for inference
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Inference: Generation of the output
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        pred = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        print(f'Instruction:\t{instruction}')
        print(f'Answer:\t{pred}')
        print('-'*20)
        question_data.loc[i, f"pred_ans"] = pred
        question_data.to_csv(save_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument("--start", type=int, default="0")
    parser.add_argument("--end", type=int, default="-1")
    parser.add_argument("--output_file", type=str, default="")
    args = parser.parse_args()
    set_seed(args.seed)
    gen_ans(args)







