import zarr
import pyn5
import shutil
import numpy as np
from pathlib import Path

zarr_root = Path("L1.zarr")
print(zarr_root.absolute())
zarr_data = zarr.open(str(zarr_root.absolute()), mode="r")
n5_root = Path("L1-2.n5")

for file_path in (zarr_root / "volumes").iterdir():
    if not file_path.name.startswith("affs") and not file_path.name.startswith("."):
        dataset_name = "volumes/{}".format(file_path.name)
        zarr_dataset = zarr_data[dataset_name]
        dtype = "{}".format(zarr_dataset.dtype).upper()
        if (n5_root / "volumes" / file_path.name).exists():
            shutil.rmtree(n5_root / "volumes" / file_path.name)
        all_data = zarr_dataset[:, :, :].transpose([2, 1, 0])
        # if dtype == "UINT64":
        #     vals = list(set(all_data.flatten()))
        #     mapping = {v: i for i, v in enumerate(vals)}
        #     fv = np.vectorize(lambda x: mapping[x])
        #     all_data = fv(all_data)
        #     dtype = "UINT64"
        pyn5.create_dataset(
            str(n5_root.absolute()),
            dataset_name,
            zarr_dataset.shape[::-1],
            [64, 64, 64],
            dtype,
        )
        n5_dataset = pyn5.open(str(n5_root.absolute()), dataset_name)
        pyn5.write(
            n5_dataset,
            (np.array([0, 0, 0]), np.array(zarr_dataset.shape[::-1])),
            all_data,
            dtype,
        )
