import math
from collections import defaultdict
import datetime as dt
import json
import os
import pickle as pkl
from typing import List

import numpy as np
import torch
from torch.utils import data
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torchvision.transforms import transforms
import zarr

from transforms import (
    Identity,
    Normalize,
    RandomSamplePixels,
    RandomSampleTimeSteps,
    ToTensor,
)
from utils import label_utils



def sigmoid(x, L ,x0, k, b):
    y = L / (1 + np.exp(-k*(x-x0)))+b
    return (y)


class PixelSetData(data.Dataset):
    def __init__(
        self,
        data_root,
        datasets,
        classes,
        transform=None,
        split=None,
        fold_num=0,
        with_extra=False,
        split_by_block=True,
    ):
        super(PixelSetData, self).__init__()

        self.transform = transform
        self.with_extra = with_extra
        self.split_by_block = split_by_block

        self.classes = classes
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

        if isinstance(datasets, str):
            datasets = [datasets]

        self.samples = []
        self.tiles = sorted(datasets)
        self.sample_tiles = []
        self.tile_to_idx = {tile: idx for idx, tile in enumerate(datasets)}
        dataset_split = json.load(open(os.path.join(data_root, 'dataset_split.json'), 'r'))
        for dataset in datasets:
            if split is not None:
                indices = dataset_split[dataset][fold_num][split]
            else:
                indices = None
            country = dataset.split("/")[-3]
            folder = os.path.join(data_root, dataset)
            data_folder = os.path.join(folder, "data")
            meta_folder = os.path.join(folder, "meta")
            samples = self.make_dataset(
                data_folder, meta_folder, self.class_to_idx, indices, country
            )
            self.samples.extend(samples)
            self.sample_tiles.extend([self.tile_to_idx[dataset]] * len(samples))


    def get_shapes(self):
        return [
            (len(dates), 10, n_pixels)
            for _, dates, n_pixels, _, _, _ in self.samples
        ]

    def get_labels(self):
        return np.array([x[3] for x in self.samples])

    def get_tiles(self):
        return self.sample_tiles

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, date_positions, n_pixels, y, extra, gdd = self.samples[index]
        pixels = zarr.load(path)  # (T, C, S)

        sample = {
            "index": index,
            "pixels": pixels,
            "valid_pixels": np.ones(
                (pixels.shape[0], pixels.shape[-1]), dtype=np.float32),
            "positions": np.array(date_positions),
            "extra": np.array(extra),
            "gdd": np.array(gdd),
            "label": y,
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def make_dataset(self, data_folder, meta_folder, class_to_idx, indices, country):
        metadata = pkl.load(open(os.path.join(meta_folder, "metadata.pkl"), "rb"))
        weather_data = pkl.load(open(os.path.join(meta_folder, "weather_data.pkl"), "rb"))

        instances = []
        code_to_class_name = label_utils.get_code_to_class(country)
        unknown_crop_codes = set()

        dates = metadata["dates"]
        date_positions = self.days_after(metadata["start_date"], dates)


        for parcel_idx, parcel in enumerate(metadata["parcels"]):
            # split data train/val/test
            if indices is not None:
                if self.split_by_block:
                    if not parcel['block'] in indices:
                        continue
                else: 
                    if not parcel_idx in indices:
                        continue

            # parse labels
            crop_code = parcel["label"]
            if country in ['austria', 'denmark']:
                if country == 'denmark' and math.isnan(crop_code):
                    crop_code = -1  # set to unknown if nan
                else:
                    crop_code = int(crop_code)
            else:
                crop_code = str(crop_code)
            parcel_path = os.path.join(data_folder, f"{parcel_idx}.zarr")
            if crop_code not in code_to_class_name:
                unknown_crop_codes.add(crop_code)
            class_name = code_to_class_name.get(crop_code, "unknown")

            if class_name not in class_to_idx:
                continue
            class_index = class_to_idx.get(class_name)

            extra = parcel['geometric_features']
            n_pixels = parcel["n_pixels"]
            t_min = weather_data['t_min'][parcel_idx]
            t_max = weather_data['t_max'][parcel_idx]

            t_base, t_cap = 0, 30
            gdd = np.maximum(
                (np.minimum(t_max, t_cap) + np.maximum(t_min, t_base)) / 2 - t_base, 0
            )
            gdd = np.cumsum(gdd, axis=0)

            # fit sigmoid to GDD

            # if t_min.mean() in lookup:
            #     gdd = deepcopy(lookup[t_min.mean()])
            # else:
            #     t_base, t_cap = 0, 30
            #     gdd = np.maximum(
            #         (np.minimum(t_max, t_cap) + np.maximum(t_min, t_base)) / 2 - t_base, 0
            #     )
            #     gdd = np.cumsum(gdd, axis=0)
            #     x = np.arange(gdd.shape[0])
            #     p0 = [max(gdd), np.median(x),1,min(gdd)]
            #     popt, _ = curve_fit(sigmoid, x, gdd, p0, method='dogbox')
            #     gdd = sigmoid(x, *popt)
            #     gdd = np.gradient(gdd)
            #     lookup[t_min.mean()] = gdd



            gdd = gdd[date_positions]

            # gdd = np.array([t_min, t_max]).T
            
            item = (parcel_path, date_positions, n_pixels, class_index, extra, gdd)
            instances.append(item)

        # for crop_code in unknown_crop_codes:
        #     print(
        #         f"Parcels with crop code {crop_code} was not found in .yml class mapping and was assigned to unknown."
        #     )

        return instances

    def days_after(self, start_date, dates):
        def parse(date):
            d = str(date)
            return int(d[:4]), int(d[4:6]), int(d[6:])

        def interval_days(date1, date2):
            return abs((dt.datetime(*parse(date1)) - dt.datetime(*parse(date2))).days)

        date_positions = [interval_days(d, start_date) for d in dates]
        return date_positions

    def get_unknown_labels(self):
        """
        Reports the categorization of crop codes for this dataset
        """
        class_count = defaultdict(int)
        class_parcel_size = defaultdict(float)
        # metadata = pkl.load(open(os.path.join(self.meta_folder, 'metadata.pkl'), 'rb'))
        metadata = self.metadata
        for meta in metadata["parcels"]:
            class_count[meta["label"]] += 1
            class_parcel_size[meta["label"]] += meta["n_pixels"]

        class_avg_parcel_size = {
            cls: total_px / class_count[cls]
            for cls, total_px in class_parcel_size.items()
        }

        code_to_class_name = label_utils.get_code_to_class(self.country)
        codification_table = label_utils.get_codification_table(self.country)
        unknown = []
        known = defaultdict(list)
        for code, count in class_count.items():
            avg_pixels = class_avg_parcel_size[code]
            if self.country == "denmark":
                code = int(code)
            code_name = codification_table[str(code)]
            if code in code_to_class_name:
                known[code_to_class_name[code]].append(
                    (code, code_name, count, avg_pixels)
                )
            else:
                unknown.append((code, code_name, count, avg_pixels))

        print("\nCategorized crop codes:")
        for class_name, codes in known.items():
            total_parcels = sum(x[2] for x in codes)
            avg_parcel_size = sum(x[3] for x in codes) / len(codes)
            print(f"{class_name} (n={total_parcels}, avg size={avg_parcel_size:.3f}):")
            codes = reversed(sorted(codes, key=lambda x: x[2]))
            for code, code_name, count, avg_pixels in codes:
                print(f"  {code}: {code_name} (n={count}, avg pixels={avg_pixels:.1f})")
        unknown = reversed(sorted(unknown, key=lambda x: x[2]))
        print("\nUncategorized crop codes:")
        for code, code_name, count, avg_pixels in unknown:
            print(f"  {code}: {code_name} (n={count}, avg pixels={avg_pixels:.1f})")


def create_train_loader(ds, batch_size, num_workers):
    return DataLoader(
        dataset=ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=torch.cuda.is_available(),
    )


def create_evaluation_loaders(datasets, config, sample_pixels_val=False):
    """
    Create data loaders for unsupervised domain adaptation
    """

    # Validation dataset
    val_transform = transforms.Compose(
        [
            RandomSamplePixels(config.num_pixels) if sample_pixels_val else Identity(),
            Normalize(),
            ToTensor(),
        ]
    )
    val_dataset = PixelSetData(
        config.data_root,
        datasets,
        config.classes,
        val_transform,
        split='val',
        fold_num=config.fold_num,
    )
    val_loader = data.DataLoader(
        val_dataset,
        num_workers=config.num_workers,
        batch_sampler=GroupByShapesBatchSampler(
            val_dataset, config.batch_size, by_pixel_dim=not sample_pixels_val, by_time=True
        ),
    )

    # Test dataset
    test_transform = transforms.Compose(
        [
            Normalize(),
            ToTensor(),
        ]
    )
    test_dataset = PixelSetData(
        config.data_root,
        datasets,
        config.classes,
        test_transform,
        split='test',
        fold_num=config.fold_num,
    )
    test_loader = data.DataLoader(
        test_dataset,
        num_workers=config.num_workers,
        batch_sampler=GroupByShapesBatchSampler(test_dataset, config.batch_size),
    )

    print(f"evaluation dataset:", datasets)
    print(f"val data: {len(val_dataset)} ({len(val_loader)} batches)")
    print(f"test data: {len(test_dataset)} ({len(test_loader)} batches)")

    return val_loader, test_loader


class GroupByShapesBatchSampler(torch.utils.data.BatchSampler):
    """
    Group parcels by their time and/or pixel dimension, allowing for batches
    with varying dimensionality.
    """

    def __init__(self, data_source, batch_size, by_time=True, by_pixel_dim=True):
        self.batches = []
        self.data_source = data_source

        shapes = data_source.get_shapes()

        # group indices by (seq_length, n_pixels)
        shp_to_indices = defaultdict(list)  # unique shape -> sample indices
        for idx, shp in enumerate(shapes):
            key = []
            if by_time:
                key.append(shp[0])
            if by_pixel_dim:
                key.append(shp[2])
            shp_to_indices[tuple(key)].append(idx)

        # create batches grouped by shape
        batches = []
        for indices in shp_to_indices.values():
            if len(indices) > batch_size:
                batches.extend(
                    [
                        indices[i : i + batch_size]
                        for i in range(0, len(indices), batch_size)
                    ]
                )
            else:
                batches.append(indices)

        self.batches = batches
        self.dataset = data_source
        self.batch_size = batch_size
        self._unit_test()

    def __iter__(self):
        for batch in self.batches:
            yield batch

    def __len__(self):
        return len(self.batches)

    def _unit_test(self):
        # make sure that we iterate across all items
        # 1) no duplicates
        assert sum(len(batch) for batch in self.batches) == len(self.dataset)
        # 2) all indices are present
        assert set([idx for indices in self.batches for idx in indices]) == set(
            range(len(self.dataset))
        )

        # make sure that no batch is larger than batch size
        assert all(len(batch) <= self.batch_size for batch in self.batches)


if __name__ == "__main__":
    # classes = label_utils.get_classes("denmark")
    classes = sorted(['corn', 'horsebeans', 'meadow', 'spring_barley', 'unknown', 'winter_barley', 'winter_rapeseed', 'winter_triticale', 'winter_wheat'])

    # crop_type = 'winter_wheat'
    # crop_type = 'spring_barley'
    # classes = [crop_type]

    from tqdm import tqdm
    tiles = ['denmark/32VNH/2017', 'austria/33UVP/2017', 'france/31TCJ/2017', 'france/30TXT/2017']
    # for tile in ['denmark/32VNH', 'france/30TXT', 'austria/33UVP']:
    for tile in ['denmark/32VNH/2017', 'denmark/32VNH/2018', 'denmark/32VNH/2019', 'denmark/32VNH/2020']:
        # dataset = PixelSetData("/media/data/timematch_l2a", [f"{tile}/2017"], classes)
        # dataset = PixelSetData("/media/data/timematch_data", [tile], classes)
        dataset = PixelSetData("/media/data/timematch_data", tiles, classes)
        print(len(dataset))

        exit()



        labels = dataset.get_labels()

        print(tile)
        for (cls, count) in zip(*np.unique(labels, return_counts=True)):
            print(classes[cls], count)
        print()
        continue

        # ndvis = []
        # gdds = []
        # positions = None
        # total_pixels, total_cloudy = 0, 0

        # for i, sample in enumerate(tqdm(dataset)):
        #     # print(sample['pixels'].max())
        #     # continue

        #     pixels = sample['pixels'] / (2**16-1)

        #     if pixels.shape[1] == 11:
        #         cloud_mask = pixels[:, 10]
        #         no_cloud_mask = cloud_mask == 0  # (T, S)

        #         total_pixels += pixels.shape[2]
        #         total_cloudy += (cloud_mask > 0).sum(1)

        #         pixels = np.moveaxis(pixels, 1, 0)  # (T, C, S) -> (C, T, S)
        #         pixels = pixels * no_cloud_mask
        #         with np.errstate(divide='ignore', invalid='ignore'):  # if only clouds, set to NaN
        #             red = pixels[2, :].sum(1) / no_cloud_mask.sum(1)
        #             nir = pixels[3, :].sum(1) / no_cloud_mask.sum(1)
        #         with np.errstate(divide='ignore', invalid='ignore'):  # if only clouds, set to NaN
        #             ndvi = (nir - red) / (nir + red)
        #         ndvis.append(ndvi)
        #         gdds.append(sample['gdd'])
        #         if positions is None:
        #             positions = sample['positions']
        #     else:
        #         red = pixels[:, 2].mean(1)
        #         nir = pixels[:, 3].mean(1)
        #         ndvi = (nir - red) / (nir + red + 1e-10)
        #         gdd = sample['gdd']
        #         np.save(f'weather_data/{tile.split("/")[-1]}_{crop_type}_gdd.npy', gdd)
        #         np.save(f'weather_data/{tile.split("/")[-1]}_{crop_type}_ndvi.npy', ndvi)
        #         np.save(f'weather_data/{tile.split("/")[-1]}_{crop_type}_pos.npy', sample['positions'])
        #     if i == 1000:
        #         break

        # ndvis = np.nanmean(np.array(ndvis), axis=0)
        # gdds = np.mean(gdds, axis=0)
        # cloudy_pct = (total_cloudy / total_pixels)*100.0
        # valid = cloudy_pct <= 80.0

        # ndvis = ndvis[valid]
        # # gdds = gdds[valid]
        # positions = positions[valid]

        # np.save(f'weather_data/{tile.split("/")[-1]}_{crop_type}_tmin_tmax.npy', gdds)
        # # np.save(f'weather_data/{tile.split("/")[-1]}_{crop_type}_gdd.npy', gdds)
        # np.save(f'weather_data/{tile.split("/")[-1]}_{crop_type}_ndvi.npy', ndvis)
        # np.save(f'weather_data/{tile.split("/")[-1]}_{crop_type}_pos.npy', positions)
        # ndvis = np.nanmean(np.array(ndvis), axis=0)

        gdds = []
        all_date_positions = []
        for i, (path, date_positions, n_pixels, y, extra, gdd) in enumerate(dataset.samples):
            gdds.append(gdd)
            all_date_positions.append(date_positions)

        gdds = np.array(gdds)
        gdds = np.unique(gdds, axis=0)
        np.save(f'weather_data/{tile.split("/")[-1]}_gdd.npy', gdds)
        all_date_positions = np.array(all_date_positions)
        all_date_positions = np.unique(all_date_positions, axis=0)
        print(list(all_date_positions[0]))
        # print(gdds.shape)

        # print(len(dataset))
