import pyn5 as pyn5
import networkx as nx
import catpy
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import gunpowder as gp
import pickle
from tqdm import tqdm

INTERP = [True, False]
CATMAID_SERVER = ""
CATMAID_API_TOKEN = ""
START = np.array((7130, 5840, 597), dtype=int)
END = np.array((8255, 6965, 677), dtype=int)
BBOX = gp.Roi(gp.Coordinate(START), gp.Coordinate(END - START))
OFFSET = np.array((0, 0, 6040), dtype=int)
RESOLUTION = np.array((3.8, 3.8, 50), dtype=float)

client = catpy.CatmaidClient(CATMAID_API_TOKEN, CATMAID_SERVER)

query = {
    "minx": START[0] * RESOLUTION[0] + OFFSET[0],
    "miny": START[1] * RESOLUTION[1] + OFFSET[1],
    "minz": START[2] * RESOLUTION[2] + OFFSET[2],
    "maxx": END[0] * RESOLUTION[0] + OFFSET[0],
    "maxy": END[1] * RESOLUTION[1] + OFFSET[1],
    "maxz": END[2] * RESOLUTION[2] + OFFSET[2],
}
skeletons = client.fetch("/1/skeletons/in-bounding-box", method="POST", data=query)

print(len(skeletons))
root = "/home/pattonw/Work/Data/n5_datasets/L1-segmented/L1.n5"
dataset = "volumes/segmentation_20"
assert Path(root).is_dir(), "root is not a directory"
assert Path(root, dataset).is_dir(), "dataset is not a directory"

image_dataset = pyn5.open(root, dataset)
image_data = image_dataset.read_ndarray((0, 0, 0), (1125, 1125, 80))

# plt.imshow(image_data[:, :, 0])
# plt.show()

for interp in INTERP:
    previous_keys = set(np.unique(image_data))

    mapping = {}
    current_id = 0
    skeleton_graph = nx.DiGraph()
    skipped = 0
    for skeleton in tqdm(skeletons):
        if current_id in previous_keys:
            skipped += 1
        while current_id in previous_keys:
            current_id += 1
        compact_detail = client.fetch(
            "/1/skeletons/{}/compact-detail".format(skeleton), method="GET"
        )
        nodes = compact_detail[0]
        node_ids, parent_ids, _, xs, ys, zs, _, _ = zip(*nodes)
        for nid, x, y, z in zip(node_ids, xs, ys, zs):
            location = (np.array((x, y, z)) - OFFSET) / RESOLUTION
            if BBOX.contains(gp.Coordinate(location)):
                skeleton_graph.add_node(nid, location=location - START, label=current_id)

        if interp:
            for nid, pid in zip(node_ids, parent_ids):
                if pid in skeleton_graph.nodes and nid in skeleton_graph.nodes:
                    a = skeleton_graph.nodes[nid]["location"]
                    b = skeleton_graph.nodes[pid]["location"]
                    slope = b - a
                    dist = np.linalg.norm(b - a)
                    for i in range(1, int(dist // 20)):
                        skeleton_graph.add_node(
                            tuple(a + 20 * slope / dist),
                            location=a + 20 * slope / dist,
                            label=current_id,
                        )
        current_id += 1

    for nid, node_data in tqdm(skeleton_graph.nodes.items()):
        location = gp.Coordinate(node_data["location"])
        old_label = image_data[tuple(location)]
        if mapping.get(old_label, None) is None:
            mapping[old_label] = set([])
        mapping[old_label].add(node_data["label"])

    print(skipped)
    print(len(mapping))
    pickle.dump(
        mapping,
        Path("mapping_{}.obj".format("interpolated" if interp else "plain")).open("wb"),
    )
