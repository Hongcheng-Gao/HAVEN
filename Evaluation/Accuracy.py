import pandas as pd
import pickle
import argparse

def Accuracy_cal(args):
    model_names = ["Valley","Qwen2.5-3B","Qwen2.5-7B","GPT4o-mini"]
    dataset_names = ["activitynet", "coin", "sports1m", "youtube"]
    results = []
    total_count = 6497

    for model_name in model_names:
        model_result = {
            "Model Name": model_name,
            "Total Correct": 0,
            "Total Correctness Rate": 0
        }

        total_correct = 0
        causes_components_count = {
            ("P", "O"): 0, ("P", "E"): 0, ("P", "S"): 0,
            ("I", "O"): 0, ("I", "E"): 0, ("I", "S"): 0,
            ("N", "O"): 0, ("N", "E"): 0, ("N", "S"): 0
        }
        causes_components_total = {
            ("P", "O"): 0, ("P", "E"): 0, ("P", "S"): 0,
            ("I", "O"): 0, ("I", "E"): 0, ("I", "S"): 0,
            ("N", "O"): 0, ("N", "E"): 0, ("N", "S"): 0
        }
        # causes_components_count = {
        #     ("P", "tf"): 0, ("P", "choice"): 0, ("P", "open"): 0,
        #     ("I", "tf"): 0, ("I", "choice"): 0, ("I", "open"): 0,
        #     ("N", "tf"): 0, ("N", "choice"): 0, ("N", "open"): 0
        # }
        # causes_components_total = {
        #     ("P", "tf"): 0, ("P", "choice"): 0, ("P", "open"): 0,
        #     ("I", "tf"): 0, ("I", "choice"): 0, ("I", "open"): 0,
        #     ("N", "tf"): 0, ("N", "choice"): 0, ("N", "open"): 0
        # }

        # causes_components_count = {
        #     ("O", "tf"): 0, ("S", "choice"): 0, ("E", "open"): 0,
        #     ("O", "tf"): 0, ("S", "choice"): 0, ("E", "open"): 0,
        #     ("O", "tf"): 0, ("S", "choice"): 0, ("E", "open"): 0
        # }
        # causes_components_total = {
        #     ("O", "tf"): 0, ("S", "choice"): 0, ("E", "open"): 0,
        #     ("O", "tf"): 0, ("S", "choice"): 0, ("E", "open"): 0,
        #     ("O", "tf"): 0, ("S", "choice"): 0, ("E", "open"): 0
        # }
        for dataset_name in dataset_names:
            file_path = args.file_path
            data = pd.read_csv(file_path, encoding='ISO-8859-1')
            data["Correctness"] = pd.to_numeric(data["Correctness"], errors="coerce")

            correct_count = data["Correctness"].sum()
            model_result[f"{dataset_name} Correct Count"] = correct_count
            model_result[f"{dataset_name} Correctness Rate"] = correct_count / len(data)
            total_correct += correct_count

            for category in ["P", "I", "N"]:
                for component in ["O", "S", "E"]:
                    category_component_data = data[(data["Causes"] == category) & (data["Components"] == component)]
                    if (category, component) not in causes_components_total:
                        causes_components_total[(category, component)] = 0
                    else:
                        causes_components_total[(category, component)] += category_component_data.shape[0]
                    correct_in_category_component = category_component_data["Correctness"].sum()
                    if (category, component) not in causes_components_count:
                        causes_components_count[(category, component)] = 0
                    else:
                        causes_components_count[(category, component)] += correct_in_category_component
        for (category, component) in causes_components_count:
            total = causes_components_total[(category, component)]
            correct = causes_components_count[(category, component)]
            correctness_rate = correct / total if total > 0 else 0
            model_result[f"{category}-{component} Correctness Rate"] = correctness_rate
        total_correct_rate = total_correct / (total_count)
        model_result["Total Correct"] = total_correct
        model_result["Total Correctness Rate"] = total_correct_rate
        results.append(model_result)
    pickle_file_path = args.pickle_path
    with open(pickle_file_path, "wb") as f:
        pickle.dump(results, f)
    print(f"Correctness rates saved to {pickle_file_path}")
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
    Accuracy_cal(args)