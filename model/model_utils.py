import os
from torch.utils.data import Dataset

import pandas as pd
from torchvision import transforms, datasets
from torchvision.transforms import v2
import torch


IMAGE_FOLDER_DIR = "/home/stoyelq/Documents/dfobot_data/small_image_folder/"
NUM_WORKERS = 4
CROP_SIZE = (800, 800)


def load_dmapps_report():
    gt_file = os.path.join("/home/stoyelq/Documents/dfobot_data/GT_metadata.csv")
    gt_data = pd.read_csv(gt_file)
    return gt_data


class my_subset(Dataset):
    r"""
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
        labels(sequence) : targets as required for the indices. will be the same length as indices
    """
    def __init__(self, dataset, indices, labels):
        self.dataset = dataset
        self.indices = indices
        labels_hold = torch.ones(len(dataset)).type(torch.long) * 300 #( some number not present in the #labels just to make sure
        labels_hold[self.indices] = labels
        self.labels = labels_hold
    def __getitem__(self, idx):
        image = self.dataset[self.indices[idx]][0]
        label = self.labels[self.indices[idx]]
        return (image, label)

    def __len__(self):
        return len(self.indices)


def get_dataloaders(batch_size, max_size=None):
    data_transforms = {
        'train': transforms.Compose([
            transforms.ToTensor(),
            v2.ToDtype(torch.uint8, scale=True),
            v2.RandomRotation(180),
            v2.RandomResizedCrop(size=CROP_SIZE),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.34, 0.39, 0.38], std=[0.1, 0.1, 0.1]),
        ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.34, 0.39, 0.38], std=[0.1, 0.1, 0.1]),
        ]),
    }

    data_dir = IMAGE_FOLDER_DIR

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    if max_size is not None:
        image_datasets['train'] = torch.utils.data.Subset(image_datasets["train"], torch.arange(max_size))
        image_datasets['val'] = torch.utils.data.Subset(image_datasets["val"], torch.arange(max_size))

    dataloaders = {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS)
        for x in ['train', 'val']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    return dataloaders, dataset_sizes, class_names



