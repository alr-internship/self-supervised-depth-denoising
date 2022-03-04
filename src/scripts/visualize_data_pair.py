
from pathlib import Path

from matplotlib import pyplot as plt
from dataset.dataset_interface import DatasetInterface
from utils.transformation_utils import to_rgb


raw_dir = Path("resources/images/raw/ycb_video")
cal_dir = Path("resources/images/calibrated/cropped/ycb_video")
raw_files = DatasetInterface.get_paths_in_dir(raw_dir)

idx = 0
raw_file = raw_files[0]
cal_file = cal_dir / raw_file.relative_to(raw_dir)
raw_rs_rgb, _, raw_zv_rgb, _, _ = DatasetInterface.load(raw_file)
cal_rs_rgb, _, cal_zv_rgb, _, _ = DatasetInterface.load(cal_file)

fig, ax = plt.subplots(2, 2, figsize=(2*16, 2*9))
# fig.tight_layout()
ax[0][0].axis('off')
ax[0][0].imshow(to_rgb(raw_rs_rgb))
ax[0][1].axis('off')
ax[0][1].imshow(to_rgb(raw_zv_rgb))
ax[1][0].axis('off')
ax[1][0].imshow(to_rgb(cal_rs_rgb))
ax[1][1].axis('off')
ax[1][1].imshow(to_rgb(cal_zv_rgb))
plt.subplots_adjust(left=0.0, bottom=0.0, right=1, top=1, wspace=0.01, hspace=0.01)
# plt.show()
plt.savefig("test.png", bbox_inches='tight', pad_inches=0)
