import os
from torch.utils.data import Dataset

import pandas as pd
from torchvision import transforms, datasets
from torchvision.transforms import v2
import torch




def get_dataloaders(batch_size, max_size=None, config_dict=None):
    NUM_WORKERS = config_dict['NUM_WORKERS']
    CROP_SIZE = config_dict['CROP_SIZE']
    VAL_CROP_SIZE = config_dict['VAL_CROP_SIZE']
    IMAGE_FOLDER_DIR = config_dict['IMAGE_FOLDER_DIR']

    data_transforms = {
        'train': transforms.Compose([
            transforms.ToTensor(),
            v2.ToDtype(torch.uint8, scale=True),
            v2.RandomRotation(180),
            v2.RandomResizedCrop(size=CROP_SIZE),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.35, 0.39, 0.37], std=[0.1, 0.11, 0.11]),

        ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(VAL_CROP_SIZE),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.35, 0.39, 0.37], std=[0.1, 0.11, 0.11]),

        ]),
    }

    data_dir = IMAGE_FOLDER_DIR

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    if max_size is not None:
        image_datasets['train'] = torch.utils.data.Subset(image_datasets["train"], torch.arange(max_size))
        image_datasets['val'] = torch.utils.data.Subset(image_datasets["val"], torch.arange(max_size))

    dataloaders = {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
        for x in ['train', 'val']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    return dataloaders, dataset_sizes, class_names



