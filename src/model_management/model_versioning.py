import os, json

META = "models/metadata.json"

def get_next_version():
    if not os.path.exists(META):
        return 1

    with open(META) as f:
        data = json.load(f)

    return data["latest_version"] + 1