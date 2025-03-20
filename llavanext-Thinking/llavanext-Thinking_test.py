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
        from HAVEN.baselines.llavanext_modeling import LLaVANeXT
        ckpt_path = "HAVEN/llavanext-Thinking/llavanext_merge_model"
        model = LLaVANeXT({"model_path": ckpt_path, "device": 0})
        return model

    dataset = args.dataset
    question_data = pd.read_csv(args.dataset_path,encoding='ISO-8859-1')
    question_data[f"pred_ans"] = None
    base_video_path = args.video_path
    choice_instruction = '''The following is a multiple-choice question. Please choose one of the following options. '''
    judge_instruction = '''Please answer the following yes/no question. '''
    cot_prompt  = ''' Give a detailed analysis step by step and provide the final answer in the last.'''
    model = load_model(TESTING_MODEL)

    start = args.start
    end = args.end
    output_file = args.output_file
    save_file = f'{output_file}/llavanext-Thinking_test.csv'
    for i in tqdm(range(start,end)):
        question = question_data.iloc[i]['Question']
        video_id = question_data.iloc[i]['video_id']
        video_path = os.path.join(base_video_path, f"{video_id}.mp4")
        if question_data.iloc[i]['Form'] == 'choice':
            instruction = choice_instruction + "\n" + question + "\n" + cot_prompt
        elif question_data.iloc[i]['Form'] == 'tf':
            instruction = judge_instruction + "\n" + question + "\n" + cot_prompt
        else:
            instruction = question + "\n" + cot_prompt

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
    parser.add_argument("--dataset_path", type=str,default='')
    parser.add_argument("--video_path", type=str,default='')
    parser.add_argument("--start", type=int, default="0")
    parser.add_argument("--end", type=int, default="-1")
    parser.add_argument("--output_file", type=str, default="")
    args = parser.parse_args()
    set_seed(args.seed)
    eval_model(args)