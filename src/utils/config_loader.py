import yaml

def load_config(path="config/system_config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)