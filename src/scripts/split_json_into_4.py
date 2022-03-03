
import json
from utils.general_utils import split


with open("bwunicluster/hp_train/adaptations.json", mode='r') as f:
    list_of_adaptions = json.load(f)

chunked_list_of_adaptions = split(list_of_adaptions, 4)

for idx, chunk_list_of_adaptions in enumerate(chunked_list_of_adaptions):
    with open(f"bwunicluster/hp_train/adaptations_{idx}.json", mode='w') as f:
        json.dump(chunk_list_of_adaptions, f)
