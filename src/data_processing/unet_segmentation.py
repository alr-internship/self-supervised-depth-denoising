from argparse import ArgumentParser
from ctypes import resize
import logging
import os
import sys
from PIL import Image
import cv2
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from torchvision import transforms
import torch.nn.functional as F
from dataset.dataset_interface import DatasetInterface
from pathlib import Path
# 3rdparty dependencies
from ycb_unet.utils.data_loading import BasicDataset
from ycb_unet.unet import UNet


ROOT_DIR = os.path.abspath(__file__ + "/../../../")


def predict_img(net,
                net_classes,
                scale,
                images,
                device,
                out_threshold=0.5):
    processed_images = []
    for image in images:
        image_region = Image.fromarray(image)
        processed_images.append(BasicDataset.preprocess(image_region, scale, is_mask=False))

    net.eval()
    img = torch.from_numpy(np.asarray(processed_images))
    # img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    masks = []
    with torch.no_grad():
        outputs = net(img)

        for image, output in zip(images, outputs):
            if net_classes > 1:
                probs = F.softmax(output, dim=1)
            else:
                probs = torch.sigmoid(output)

            tf = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(image.shape[:2]),
                transforms.ToTensor()
            ])

            full_mask = tf(probs.cpu()).squeeze()

            if net_classes == 1:
                masks.append((full_mask > out_threshold).numpy())
            else:
                masks.append(F.one_hot(full_mask.argmax(dim=0), net_classes).permute(2, 0, 1).numpy())
    
    return masks


def compute_bound(rgb_image):
    indices = (rgb_image != [0, 0, 0]).nonzero()
    mins = np.min(indices, axis=1)[:2]
    maxs = np.max(indices, axis=1)[:2] + [1, 1]  # max bound exclusive
    return mins, maxs


def main(args):
    input_dir = args.input_dir
    output_dir = args.output_dir
    paths = DatasetInterface.get_paths_in_dir(input_dir, recursive=True)
    assert paths != 0, f"{input_dir} does not contain any data to mask"

    # Directory to save logs and trained model
    model_file = Path(ROOT_DIR) / "3rdparty/ycb_unet/resources/1645045183.913909/checkpoint_epoch2.pth"
    use_zivid_rgbs = False
    batch_size = 10
    scale = 0.5
    mask_threshold = 0.5

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {model_file}')
    logging.info(f'Using device {device}')

    net_classes = 2
    net = UNet(n_channels=3, n_classes=net_classes)
    net = nn.DataParallel(net)

    net.to(device=device)
    net.load_state_dict(torch.load(model_file, map_location=device))

    logging.info('Model loaded!')

    paths_chunked = list(zip(*[iter(paths)]*batch_size))
    print("chuncked paths:", len(paths_chunked))

    for paths_chunk in tqdm(paths_chunked):
        images_tuples = []
        images = []
        images_bounds = []
        for path in paths_chunk:
            rs_rgb, rs_depth, zv_rgb, zv_depth, _ = DatasetInterface.load(path)
            images_tuples.append((rs_rgb, rs_depth, zv_rgb, zv_depth))
            zv_rgb = cv2.cvtColor(zv_rgb, cv2.COLOR_BGR2RGB)

            # resize region of image with no nans to original size to get best prediction results
            image = zv_rgb if use_zivid_rgbs else rs_rgb
            min_bound, max_bound = compute_bound(image)
            image_region = image[min_bound[0]:max_bound[0], min_bound[1]:max_bound[1]]
            image_region = cv2.resize(image_region, tuple(reversed(image.shape[:2])))
            if len(image_region.shape) == 2:
                image_region = image_region[..., None]
            images.append(image_region)
            images_bounds.append((min_bound, max_bound))

        masks_list = predict_img(net=net,
                            net_classes=net_classes,
                            scale=scale,
                            images=images,
                            out_threshold=mask_threshold,
                            device=device)

        tqdm.write("save images with computed mask")
        for image_bounds, path, images_tuples, masks in zip(images_bounds, paths_chunk,
                                                             images_tuples, masks_list):
            # rel_path = dataset_interface.data_file_paths[idx].relative_to(input_dir)

            # visualize.display_instances(original_image,
            # r['rois'], r['masks'], r['class_ids'], dataset.class_names, r['scores'])
            # masks = result['masks']
            # resize mask to original image
            min_bound, max_bound = image_bounds
            size = max_bound - min_bound
            masks = masks[1].astype(np.uint16) # foreground mask is important only
            resized_masks = cv2.resize(masks, tuple(reversed(size)), interpolation=cv2.INTER_NEAREST)
            if len(resized_masks.shape) == 2:  # channel dimension got lost during cv2.resize, if only 1 wide
                resized_masks = resized_masks[..., None]
            resized_masks = resized_masks.astype(np.bool)

            total_masks = np.zeros(tuple(images_tuples[0].shape[:2]) + (resized_masks.shape[-1],), dtype=np.bool)
            total_masks[min_bound[0]:max_bound[0], min_bound[1]:max_bound[1]] = resized_masks

            rel_file_path = path.relative_to(input_dir)
            DatasetInterface.save(*images_tuples, total_masks, file_name=output_dir / rel_file_path)


if __name__ == "__main__":
    argparse = ArgumentParser()
    argparse.add_argument("input_dir", type=Path, help="dataset the mask should be computed for")
    argparse.add_argument("output_dir", type=Path, help="dataset the masked data should be saved to")
    argparse.add_argument("--gpus", type=int, default=1, help="how many gpus to use for inference")
    argparse.add_argument("--imgs-per-gpu", type=int, default=1, help="how many imgs to infer at once per gpu")
    argparse.add_argument("--use-zivids-rgbs", action='store_true',
                          help="set wich rgb image to use for ycb segmentation")
    main(argparse.parse_args())
# %%
