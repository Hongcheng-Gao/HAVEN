import base64
import requests
import json
import pandas as pd
import argparse
from tqdm import tqdm
import os
from transformers import set_seed
def GPT_eval(args):
    start_idx = args.start
    end_idx = args.end if args.end is not None else len(df)
    # Data Import
    dataset = args.dataset
    TESTING_MODEL=args.model_name
    output_file = args.output_file
    if not os.path.exists(output_file):
        os.makedirs(output_file)
    save_path = f'{output_file}/COT_Grade_{TESTING_MODEL}_{dataset}_{start_idx}.csv'
    answer_path = args.answer_file
    df = pd.read_csv(answer_path, encoding='ISO-8859-1')
    answer = df['Answer']
    res = df['pred_ans']
    q_form = df['Form']
    question = df['Question']

    # OpenAI API Key
    api_key = args.api_key
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    for idx in tqdm(range(start_idx, end_idx)):
        if q_form[idx] == 'choice':
            prompt = f'''This is a multiple-choice question. Based on the given question and reasoning process, extract the corresponding answer of the reasoning process.\n
                        Question: {question[idx]} \n                  
                        Reasoning Process: {res[idx]} \n                                                   
                        Instructions: \n
                        1. Identify the correct answer based on the reasoning process.\n
                        2. If the reasoning process directly mentions one of the given choices (A, B, or C), return the corresponding letter along with the full text of that option (e.g., "A. Option text").\n
                        3. If the reasoning process provides an answer that does not explicitly mention A, B, or C, compare its meaning to the given choices and return the best-matching option in the format "Letter. Option text".\n
                        4. If the reasoning process concludes that the correct answer is "no answer" or "I don't know", return "no answer".\n
                        5. Return only the final answer, without explanation or additional text.\n
                        6. Foucs more on the final summary sentence.\n
                        '''
            payload_1 = {
                "model": "gpt-4o-mini",
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": 30
            }
            extract_answer_2 = requests.post(args.GPT_url,headers=headers, json=payload_1).json()
            try:
                extract_answer = extract_answer_2["choices"][0]["message"]["content"].strip()
            except KeyError:
                extract_answer = "F"
            df.loc[idx, f"Extract Answer"] = extract_answer
            prompt_v = f''' 
                You are a professional homework grading tool. I will provide you with four rules for grading: \n 
                    1. This is a multiple-choice question. Judge the correctness based on the selected lettera and actual content of the provided answers. \n 
                    2. Regardless of the question type, respond only with either 1 or 0, without any additional explanation. \n  
                    3. 1 means the prediction is correct, and 0 means it is incorrect. \n 
                    4. If the predicted answer matches the correct answer in meaning, even if it is phrased differently, consider it correct. \n 
                    For example, if the prediction conveys the same meaning as the standard answer, you should respond with 1. \n 
                    Based on the question and its standard answer, is the prediction correct? If yes, return only 1; otherwise, return only 0. \n 
                    Question: {question[idx]} \n
                    Standard Answer: {answer[idx]} \n 
                    The Predicted Answer: {extract_answer} \n\n '''
                
            payload_2 = {
                "model": "gpt-4o-mini",
                "messages": [
                    {
                        "role": "user",
                        "content": prompt_v
                    }
                ],
                "max_tokens": 10
            }

            correct_ori = requests.post(args.GPT_url, headers=headers, json=payload_2)
            correct_pro = correct_ori.json() 
            try:
                correctness = correct_pro["choices"][0]["message"]["content"].strip()
            except KeyError:
                correctness = "F"
            print(f"Index: {idx}, Standard Answer: {answer[idx]}, Predicted Answer: {res[idx]}, Extracted answer: {extract_answer}, Correct: {correctness}")            
            df.loc[idx, f"Correctness"] = correctness
            df.to_csv(save_path, index=False)

        elif q_form[idx] == 'tf':
            prompt = f'''This is a multiple-choice question. Based on the given question and reasoning process, extract the corresponding answer of the reasoning process.\n
                        Question: {question[idx]} \n                  
                        Reasoning Process: {res[idx]} \n                                                   
                        Instructions: \n
                        1. Identify the correct answer based on the reasoning process.\n
                        2. If the reasoning process explicitly states "yes" or "no", return direct "yes" or "no" \n
                        3. If the reasoning process concludes that the correct answer is "no answer" or "I don't know", return "no answer".\n
                        5. Return only the final "yes" or "no", without explanation or additional text.\n
                        '''
            payload_1 = {
                "model": "gpt-4o-mini",
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": 30
            }
            extract_answer_2 = requests.post(args.GPT_url,headers=headers, json=payload_1).json()
            try:
                extract_answer = extract_answer_2["choices"][0]["message"]["content"].strip()
            except KeyError:
                extract_answer = "F"
            df.loc[idx, f"Extract Answer"] = extract_answer   
            prompt_v = f''' 
                You are a professional homework grading tool. I will provide you with four rules for grading\n
                    1. This is a yes/no question. The Standard Answer is only Yes/No, please directly compare the standard answer with 'yes' or 'no' in the predicted answer.\n
                    2. No matter what kind of questions, only response with one of 1 or 0. No more explaination. \n
                    3. 1 means correct (they are the same), 0 means wrong (they are different). \n
                    For example, if the prediction conveys the same meaning as the standard answer, you should respond with 1.\n 
                    Based on the question, is the prediction correct? If yes, only return 1, otherwise only return 0.\n\n
                    Question: {question[idx]} \n
                    Standard Answer: {answer[idx]}\n
                    The Predicted Answer: {extract_answer}\n'''
            
            payload_2 = {
                "model": "gpt-4o-mini",
                "messages": [
                    {
                        "role": "user",
                        "content": prompt_v
                    }
                ],
                "max_tokens": 10 
            }

            correct_ori = requests.post(args.GPT_url, headers=headers, json=payload_2)
            correct_pro = correct_ori.json() 
            try:
                correctness = correct_pro["choices"][0]["message"]["content"].strip()
            except KeyError:
                correctness = "F"
            print(f"Index: {idx}, Standard Answer: {answer[idx]}, Predicted Answer: {res[idx]}, Extracted answer: {extract_answer}, Correct: {correctness}")
            df.loc[idx, f"Correctness"] = correctness
            df.to_csv(save_path, index=False)

        else:
            prompt_v = f''' 
                You are a professional homework grading tool. I will provide you with four rules for grading\n
                    1. The Standard Answer is a sentence. Compare the provided predicted answers based on their meaning rather than exact wording. The prediction is correct if it conveys the same intent.\n  
                    2. If the question asks about identity or species, the predicted answer is correct as long as the core identity or species is correct, even if descriptive adjectives differ.\n
                    3. If the question asks about a scene, the predicted answer is correct if the described scene exists in the standard answer or is similar.\n  
                    4. If the answer indicates that the asked character, object or event is not visible or does not exist, it should be considered as "No answer." \n
                    5. Regardless of the question type, respond only with either 1 or 0, without any additional explanation.\n
                    6. 1 means correct, and 0 means incorrect.\n
                    For example, if the prediction conveys the same meaning as the standard answer, even if phrased differently, you should respond with 1.\n  
                    Based on the question, is the prediction correct? If yes, return only 1; otherwise, return only 0. \n\n
                    Question: {question[idx]} \n
                    Standard Answer: {answer[idx]}\n
                    The Predicted Answer: {res[idx]}\n'''
            
            payload_2 = {
                "model": "gpt-4o-mini",
                "messages": [
                    {
                        "role": "user",
                        "content": prompt_v
                    }
                ],
                "max_tokens": 10
            }

            correct_ori = requests.post(args.GPT_url, headers=headers, json=payload_2)

            correct_pro = correct_ori.json() 

            try:
                correctness = correct_pro["choices"][0]["message"]["content"].strip()
            except KeyError:
                correctness = "F"
            extract_answer = ""

            print(f"Index: {idx}, Standard Answer: {answer[idx]}, Predicted Answer: {res[idx]}, Correct: {correctness}")
            df.loc[idx, f"Extract Answer"] = extract_answer
            df.loc[idx, f"Correctness"] = correctness
            df.to_csv(save_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_name", type=str, default="LLaVA-NeXT-Video")
    parser.add_argument("--dataset", type=str, default="coin", choices=["sports1m", "activitynet", "coin", "youtube"])
    parser.add_argument("--output_file", type=str, default="")
    parser.add_argument("--answer_file", type=str, default="")
    parser.add_argument("--api_key", type=str, default="")
    parser.add_argument("--GPT_url", type=str, default="")
    parser.add_argument("--start", type=int, default=0, help="Start index of processing")
    parser.add_argument("--end", type=int, default=None, help="End index of processing (exclusive)")
    parser.add_argument("--cot", type=str, default="COT", help="Set this to cot for evaluating model with COT")
    args = parser.parse_args()
    
    set_seed(args.seed)
    GPT_eval(args)
