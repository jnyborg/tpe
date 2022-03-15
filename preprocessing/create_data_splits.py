import math
import os
import random
import numpy as np
from utils import label_utils
import glob
from collections import defaultdict
import pickle as pkl
import json

def main(seed=0):
    random.seed(seed)
    np.random.seed(seed)

    data_root = '/media/data/timematch_data'
    tiles_in_dir = sorted(list(glob.glob(f'{data_root}/*/*/*')))
    num_blocks = 100
    num_folds = 3

    tiles = defaultdict(list)
    for path in tiles_in_dir:
        country, tile, year = path.split('/')[-3:]
        tiles[f'{country}/{tile}'].append(year)


    data_splits = {}
    data_classes = {}

    classes_all = []

    for country_tile, years in tiles.items():
        country = country_tile.split("/")[0]
        # ensure same split across different years
        folds = create_train_val_test_folds(num_folds, num_blocks, val_ratio=0.1, test_ratio=0.2)
        for year in years:
            dataset_name = f'{country_tile}/{year}'
            dataset_path = os.path.join(data_root, country_tile, year)

            # select classes that appear at least 200 times
            classes = get_frequent_classes(dataset_path, country, min_count=200)
            classes_all.append(set(classes))

            data_splits[dataset_name] = folds
            data_classes[dataset_name] = classes
    print(set.intersection(*classes_all))

    json.dump(data_classes, open(os.path.join(data_root, 'dataset_classes.json'), 'w'))
    json.dump(data_splits, open(os.path.join(data_root, 'dataset_split.json'), 'w'))

    print(f'Done, result saved to {data_root}')



def get_frequent_classes(dataset_path, country, min_count=200):
    meta_folder = os.path.join(dataset_path, "meta")
    metadata = pkl.load(open(os.path.join(meta_folder, "metadata.pkl"), "rb"))

    classes = label_utils.get_classes(country)
    crop_code_to_class = label_utils.get_code_to_class(country)
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

    print(dataset_path)
    labels = []
    for parcel in metadata["parcels"]:
        crop_code = parcel["label"]
        if country in ['austria', 'denmark']:
            if country == 'denmark' and math.isnan(crop_code):
                crop_code = -1
            else:
                crop_code = int(crop_code)
        else:
            crop_code = str(crop_code)
        class_name = crop_code_to_class.get(crop_code, "unknown")
        class_index = class_to_idx.get(class_name, class_to_idx["unknown"])
        labels.append(class_index)
        # labels.append(crop_code)

    unique_classes, counts = np.unique(labels, return_counts=True)
    for cls, cnt in zip(unique_classes, counts):
        # if cnt > 200:
        #     print(cls, cnt)
        print(classes[cls], cnt)
    print()
    valid_classes = [classes[i] for i in unique_classes[counts >= min_count]]
    return valid_classes


def create_train_val_test_folds( num_folds, num_indices, val_ratio=0.1, test_ratio=0.2):
    folds = []
    for _ in range(num_folds):
        splits = {}
        indices = list(range(num_indices))
        n = len(indices)
        n_test = int(test_ratio * n)
        n_val = int(val_ratio * n)
        n_train = n - n_test - n_val

        random.shuffle(indices)

        train_indices = set(indices[:n_train])
        val_indices = set(indices[n_train : n_train + n_val])
        test_indices = set(indices[-n_test:])
        assert set.intersection(train_indices, val_indices, test_indices) == set()
        assert len(train_indices) + len(val_indices) + len(test_indices) == n

        splits = {
            "train": list(train_indices),
            "val": list(val_indices),
            "test": list(test_indices),
        }
        folds.append(splits)
    return folds

if __name__ == "__main__":
    main()
    
