from pathlib import Path
import shutil

import pandas as pd


ROOT_DIR = Path(__file__).parent.parent.parent.parent

with open(ROOT_DIR / "wandb.csv", 'r') as f:
    csv_file = pd.read_csv(f, dtype=str)

trainer_ids = csv_file['trainer_id'].to_list()

models_dir = [d for d in (ROOT_DIR / "local_resources/hp_models").iterdir() if d.is_dir()]

removed_count = 0
for model_dir in models_dir:
    if model_dir.name not in trainer_ids:
        print("removing", model_dir.name)
        removed_count += 1
        shutil.rmtree(model_dir)
    else:
        print("keeping", model_dir.name)
        trainer_ids.remove(model_dir.name)

print("wandb but not local", trainer_ids)
print(f"removed {removed_count} model dirs from total {len(models_dir)}")

