import json
import os

def log_experiment(params, metrics, file_path="outputs/experiments.json"):
    experiment = {
        "params": params,
        "metrics": metrics
    }

    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            data = json.load(f)
    else:
        data = []

    data.append(experiment)

    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)

    print("📊 Experiment logged")