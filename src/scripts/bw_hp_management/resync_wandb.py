from pathlib import Path
import shutil

import pandas as pd


ROOT_DIR = Path(__file__).parent.parent.parent.parent

with open(ROOT_DIR / "wandb_export.csv", 'r') as f:
    csv_file = pd.read_csv(f, dtype=str)

trainer_ids = csv_file['trainer_id'].to_list()

models_dir = ROOT_DIR / "local_resources/hp_models"

for model_dir in models_dir.iterdir():
    if not model_dir.is_dir():
        continue

    if model_dir.name not in trainer_ids:
        print("removing", model_dir.name)
        shutil.rmtree(model_dir)
    else:
        print("keeping", model_dir.name)
        trainer_ids.remove(model_dir.name)

print("wandb but not local", trainer_ids)

