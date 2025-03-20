import pandas as pd
import pickle
import argparse
def Consistency_cal(args):
    dataset_names = ["activitynet", "coin", "sports1m", "youtube"]
    model_names = [
        "LLaVA",
        "VideoLLaVA",
        "VideoChat2",
        "ShareGPT4Video",
        "VideoChatGPT",
        "Video-LLaMA-2",
        "Video-LLaMA-2-13B",
        "LLaMA-VID",
        "LLaMA-VID-13B",
        "PLLaVA",
        "PLLaVA-13B",
        "COT_llamavid",
        "COT_PLLaVA",
        "GPT4o-miniFIX",
        "llavanext_DPO",
        "llavanext",
        "llavanext34B",
        "llavanextNoCOT",
        "Qwen2.5-3B",
        "Qwen2.5-7B","Valley"
    ]
    results = []
    from collections import Counter
    for model_name in model_names:
        model_result = {
            "Model Name": model_name,
            "Total Consistency Rate": 0
        }

        total_q_ids_count = 0
        consistency_count = 0

        # causes_components_consistency = {
        #     ("P", "O"): 0, ("P", "E"): 0, ("P", "S"): 0,
        #     ("I", "O"): 0, ("I", "E"): 0, ("I", "S"): 0,
        #     ("N", "O"): 0, ("N", "E"): 0, ("N", "S"): 0
        # }
        # q_id_counts = {
        #     ("P", "O"): 0, ("P", "E"): 0, ("P", "S"): 0,
        #     ("I", "O"): 0, ("I", "E"): 0, ("I", "S"): 0,
        #     ("N", "O"): 0, ("N", "E"): 0, ("N", "S"): 0
        # }

        causes_components_consistency = {
            ("tf"): 0, ("choice"): 0, ("open"): 0,
            ("tf"): 0, ("choice"): 0, ("open"): 0,
            ("tf"): 0, ("choice"): 0, ("open"): 0
        }
        q_id_counts = {
            ("tf"): 0, ("choice"): 0, ("open"): 0,
            ("tf"): 0, ("choice"): 0, ("open"): 0,
            ("tf"): 0, ("choice"): 0, ("open"): 0
        }
        for dataset_name in dataset_names:
            file_path = args.file_path
            data = pd.read_csv(file_path, encoding='ISO-8859-1')
            q_id_groups = data.groupby("q_id")["Correctness"].agg(lambda x: x.nunique())
            q_id_count = Counter(data["q_id"])
            for q_id, num_unique_correctness in q_id_groups.items():
                if q_id_count[q_id] > 1:
                    total_q_ids_count += 1
                    if num_unique_correctness == 1:
                        consistency_count += 1
            for component in ["tf", "choice", "open"]:
                category_component_data = data[(data["Form"] == component)]
                category_component_q_id_groups = category_component_data.groupby("q_id")["Correctness"].agg(lambda x: x.nunique())
                for q_id, num_unique_correctness in category_component_q_id_groups.items():
                    if q_id_count[q_id] > 1:
                        q_id_counts[(component)] += 1
                        if num_unique_correctness == 1:
                            causes_components_consistency[(component)] += 1
        total_consistency_rate = consistency_count / total_q_ids_count if total_q_ids_count > 0 else 0
        model_result["Total Consistency Rate"] = total_consistency_rate
        for component in causes_components_consistency:
            model_result[f"Consistency {component} Count"] = causes_components_consistency[(component)]
            consistency_rate = causes_components_consistency[(component)] / q_id_counts[(component)] if q_id_counts[(component)] > 0 else 0
            model_result[f"Consistency Rate {component}"] = consistency_rate
        model_result["Total Consistency Count"] = consistency_count
        model_result["Total QID Count"] = total_q_ids_count
        results.append(model_result)
    pickle_file_path = args.pickle_path
    with open(pickle_file_path, "wb") as f:
        pickle.dump(results, f)
    print(f"Consistency statistics saved to {pickle_file_path}")
    with open(pickle_file_path, "rb") as f:
        loaded_results = pickle.load(f)
    df = pd.DataFrame(loaded_results)
    print(df.head())
    df.to_csv(args.save_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str,default="")
    parser.add_argument("--pickle_path", type=str, default="")
    parser.add_argument("--save_path", type=str, default="")
    args = parser.parse_args()
    Consistency_cal(args)
