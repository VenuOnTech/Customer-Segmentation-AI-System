import json
import os
from datetime import datetime

def log_data_lineage(data_version, output_path):

    lineage = {
        "timestamp": datetime.utcnow().isoformat(),
        "data_version": data_version,
        "output_file": output_path
    }

    os.makedirs("outputs", exist_ok=True)

    with open("outputs/data_lineage.json", "w") as f:
        json.dump(lineage, f, indent=4)

    print("Data lineage logged")