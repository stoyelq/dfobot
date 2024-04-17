import os

from torch import nn, optim
from torch.nn.functional import relu
from torch.utils.data import Dataset

import pandas as pd
from PIL import Image
from pathlib import Path
from torchvision import transforms, datasets, models
from torchvision.transforms import v2
import torch


METADATA_CSV_PATH = "/home/stoyelq/Documents/dfobot_data/metadata/metadata.csv"
METADATA_COLUMNS = ['age', 'length', 'weight', 'month', 'is_male',
       'is_female', 'is_unknown', 'is_plaice', 'is_herring']

class ImageFolderCustom(Dataset):
    def __init__(self, targ_dir, transform=None):
        self.paths = list(Path(targ_dir).rglob("*.jpg"))
        self.transform = transform
        self.metadata_df = pd.read_csv(METADATA_CSV_PATH)


    def get_metadata(self, path):
        # make sure this function returns the label from the path
        uuid = path.name.split("__")[0]
        metadata_row = self.metadata_df[(self.metadata_df["uuid"] == uuid)]
        out_tensor = torch.tensor(metadata_row[METADATA_COLUMNS].values[0])
        age = torch.tensor(float(metadata_row["age"].values))
        return out_tensor, age

    def load_image(self, index):
        image_path = self.paths[index]
        return Image.open(image_path)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        img = self.load_image(index)
        metadata, age = self.get_metadata(self.paths[index])

        if self.transform:
            return self.transform(img), metadata, age
        else:
            return img, metadata, age


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

    image_datasets = {x: ImageFolderCustom(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}

    if max_size is not None:
        image_datasets['train'] = torch.utils.data.Subset(image_datasets["train"], torch.arange(max_size))
        image_datasets['val'] = torch.utils.data.Subset(image_datasets["val"], torch.arange(max_size))

    dataloaders = {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
        for x in ['train', 'val']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    return dataloaders, dataset_sizes

def get_old_base_model(device, all_layers):
    model_conv = models.resnet50(weights='IMAGENET1K_V2')
    model_conv.to(device)
    # freeze inner layers, if called for:
    for param in model_conv.parameters():
        param.requires_grad = all_layers

    # swap out final fc layer:
    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_ftrs, 1)  # go to single value
    return model_conv


def get_augmented_model(device, all_layers):
    model_conv = AugmentedModel(all_layers)
    model_conv.to(device)
    return model_conv


class BaseModel(nn.Module):
    def __init__(self, all_layers):
        super(BaseModel, self).__init__()
        self.cnn = models.resnet50(weights='IMAGENET1K_V2')

        # freeze inner layers, if called for:
        for param in self.cnn.parameters():
            param.requires_grad = all_layers

        self.cnn.fc = nn.Linear(
            self.cnn.fc.in_features, 1)

    def forward(self, image, data=None):
        x = self.cnn(image)
        return x


class AugmentedModel(nn.Module):
    cnn_out_size = 50
    hidden_layer_size = 50
    meta_data_length = len(METADATA_COLUMNS)

    def __init__(self, all_layers):
        super(AugmentedModel, self).__init__()
        self.cnn = models.resnet50(weights='IMAGENET1K_V2')
        # freeze inner layers, if called for:
        for param in self.cnn.parameters():
            param.requires_grad = all_layers

        self.cnn.fc = nn.Linear(
            self.cnn.fc.in_features, self.cnn_out_size)

        self.fc1 = nn.Linear(self.cnn_out_size + self.meta_data_length, self.hidden_layer_size)
        self.fc2 = nn.Linear(self.hidden_layer_size, 1) # 1 = age

    def forward(self, image, metadata):
        cnn_out = self.cnn(image)
        cnn_out_augmented = torch.cat((cnn_out, metadata.type(cnn_out.dtype)), dim=1)
        fc1_out = relu(self.fc1(cnn_out_augmented))
        age = self.fc2(fc1_out)
        return age

def get_base_model(device, all_layers):
    model_conv = BaseModel(all_layers)
    model_conv.to(device)
    return model_conv