import pickle
import pyn5
import numpy as np
from tqdm import tqdm
import shutil
from pathlib import Path

INTERP = [True, False]
for interp in INTERP:
    root = "/home/pattonw/Work/Data/n5_datasets/L1-segmented/L1.n5"
    dataset = "volumes/segmentation_20"
    out_dataset = "volumes/segmentation_20_skeleton_mapped_{}".format(
        "interp" if interp else "no_interp"
    )

    mapping = pickle.load(
        open("mapping_{}.obj".format("interpolated" if interp else "plain"), "rb")
    )

    image_dataset = pyn5.open(root, dataset)
    image_data = image_dataset.read_ndarray((0, 0, 0), (1125, 1125, 80))

    for key, value in tqdm(mapping.items()):
        if len(value) == 1:
            image_data = np.where(image_data == key, value.pop(), image_data)
    try:
        shutil.rmtree(Path(root, out_dataset))
    except Exception as e:
        print(e)
        pass
    pyn5.create_dataset(root, out_dataset, (1125, 1125, 80), (125, 125, 10), "UINT32")
    out_ds = pyn5.open(root, out_dataset)
    out_ds.write_ndarray(np.array((0, 0, 0)), image_data, 0)

