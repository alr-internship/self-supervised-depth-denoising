# %%
from argparse import ArgumentParser
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import sys

ROOT_DIR = os.path.abspath(__file__ + "/../../../")
SRC_DIR = os.path.join(ROOT_DIR, "src")

sys.path.append(SRC_DIR)

from dataset.dataset_interface import DatasetInterface
from pathlib import Path
import numpy as np

# %% [markdown]
# # Mask

# %%
THIRDPARTY_DIR = os.path.join(ROOT_DIR, "3rdparty/mask_rcnn")
sys.path.append(THIRDPARTY_DIR)  # To find local version of the library

import mrcnn.model as modellib
from mrcnn import visualize, config
from samples.ycb.ycb_config import YCBConfig
from samples.ycb.ycb_dataset import YCBDataset

def main(args):
    # Directory to save logs and trained model
    MODEL_DIR = os.path.join(ROOT_DIR, "resources/networks/mask_rcnn/logs")
    model_file = Path(THIRDPARTY_DIR + "/resources/ycb/mask_rcnn_ycb_video_dataset_0008.h5")

    config = YCBConfig(gpus=1, imgs_per_gpu=1)
    config.display()

    model = modellib.MaskRCNN(mode='inference', model_dir=MODEL_DIR, config=config)
    model.load_weights(model_file.as_posix(), by_name=True)


    # %%
    dataset = YCBDataset()
    dataset.load_classes(THIRDPARTY_DIR + "/samples/ycb/data/YCB_Video_Dataset/annotations/val_instances.json")
    dataset.prepare()

    # %%
    input_path = Path(ROOT_DIR) / args.input_dir
    dataset_interface = DatasetInterface(input_path)
    masked_dataset_interface = DatasetInterface(Path(ROOT_DIR) / args.output_dir)

    for idx, (rs_rgb, rs_depth, zv_rgb, zv_depth) in enumerate(dataset_interface):
        print(f"masking id: {idx}")

        original_image = zv_rgb
        results = model.detect([original_image])
        r = results[0]

        rel_path = dataset_interface.data_file_paths[idx].relative_to(input_path)

        # visualize.display_instances(original_image, 
        # r['rois'], r['masks'], r['class_ids'], dataset.class_names, r['scores'])
        mask = np.expand_dims(np.sum(r['masks'], axis=2) > 0, axis=2)
        # rs_rgb_masked = rs_rgb * mask
        # rs_depth_masked = np.expand_dims(rs_depth, axis=2) * mask
        # zv_rgb_masked = zv_rgb * mask
        # zv_depth_masked = np.expand_dims(zv_depth, axis=2) * mask
        masked_dataset_interface.append_with_mask_and_save(rs_rgb, rs_depth, zv_rgb, zv_depth, mask, rel_path)


if __name__ == "__main__":
    argparse = ArgumentParser()
    argparse.add_argument("input_dir", type=str)
    argparse.add_argument("output_dir", type=str)
    main(argparse.parse_args())