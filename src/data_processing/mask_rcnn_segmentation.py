# %%
from argparse import ArgumentParser
import os
import cv2
from tqdm import tqdm

os.environ["KERAS_BACKEND"] = "tensorflow"
import sys

ROOT_DIR = os.path.abspath(__file__ + "/../../../")
SRC_DIR = os.path.join(ROOT_DIR, "src")

sys.path.append(SRC_DIR)

from dataset.dataset_interface import DatasetInterface
from pathlib import Path

# %% [markdown]
# # Mask

# %%
THIRDPARTY_DIR = os.path.join(ROOT_DIR, "3rdparty/mask_rcnn")
sys.path.append(THIRDPARTY_DIR)  # To find local version of the library

import mrcnn.model as modellib
from samples.ycb.ycb_config import YCBConfig
from samples.ycb.ycb_dataset import YCBDataset

def main(args):
    # Directory to save logs and trained model
    MODEL_DIR = os.path.join(ROOT_DIR, "resources/networks/mask_rcnn/logs")
    model_file = Path(THIRDPARTY_DIR + "/resources/ycb/mask_rcnn_ycb_video_dataset_0100.h5")
    use_zivid_rgbs = False


    gpus = args.gpus
    imgs_per_gpu = args.imgs_per_gpu
    batch_size = gpus * imgs_per_gpu

    config = YCBConfig(gpus=gpus, imgs_per_gpu=imgs_per_gpu)
    config.display()

    model = modellib.MaskRCNN(mode='inference', model_dir=MODEL_DIR, config=config)
    model.load_weights(model_file.as_posix(), by_name=True)


    # %%
    dataset = YCBDataset()
    dataset.load_classes(THIRDPARTY_DIR + "/samples/ycb/data/YCB_Video_Dataset/annotations/val_instances.json")
    dataset.prepare()

    # %%
    input_dir = args.input_dir
    output_dir = args.output_dir
    paths = DatasetInterface.get_paths_in_dir(input_dir, recursive=True)
    assert paths != 0, f"{input_dir} does not contain any data to mask"

    paths_chunked = list(zip(*[iter(paths)]*batch_size))
    print("chuncked paths:", len(paths_chunked))

    # masked_dataset_interface = DatasetInterface(Path(ROOT_DIR) / args.output_dir)

    for paths_chunk in tqdm(paths_chunked):
        images_tuples = []
        images = []
        for path in paths_chunk:
            rs_rgb, rs_depth, zv_rgb, zv_depth, _ = DatasetInterface.load(path)
            images_tuples.append((rs_rgb, rs_depth, zv_rgb, zv_depth))
            zv_rgb = cv2.cvtColor(zv_rgb, cv2.COLOR_BGR2RGB)
            if use_zivid_rgbs:
                images.append(zv_rgb)
            else:
                images.append(rs_rgb)
        
        tqdm.write("infer images")
        results = model.detect(images)

        tqdm.write("save images with computed mask")
        for path, images_tuples, result in zip(paths_chunk, images_tuples, results):
            # rel_path = dataset_interface.data_file_paths[idx].relative_to(input_dir)

            # visualize.display_instances(original_image, 
            # r['rois'], r['masks'], r['class_ids'], dataset.class_names, r['scores'])
            masks = result['masks']
            rel_file_path = path.relative_to(input_dir)
            DatasetInterface.save(*images_tuples, masks, file_name=output_dir / rel_file_path)


if __name__ == "__main__":
    argparse = ArgumentParser()
    argparse.add_argument("input_dir", type=Path, help="dataset the mask should be computed for")
    argparse.add_argument("output_dir", type=Path, help="dataset the masked data should be saved to")
    argparse.add_argument("--gpus", type=int, default=1, help="how many gpus to use for inference")
    argparse.add_argument("--imgs-per-gpu", type=int, default=1, help="how many imgs to infer at once per gpu")
    argparse.add_argument("--use-zivids-rgbs", action='store_true', help="set wich rgb image to use for ycb segmentation")
    main(argparse.parse_args())