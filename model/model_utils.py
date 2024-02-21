import os
import time
from tempfile import TemporaryDirectory

import pandas as pd
from matplotlib import pyplot as plt
from torchvision import transforms, datasets
from torchvision.transforms import v2
import torch


IMAGE_FOLDER_DIR = "/home/stoyelq/Documents/dfobot_data/image_folder/"
BATCH_SZIE = 4
NUM_WORKERS = 4
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_dmapps_report():
    gt_file = os.path.join("/home/stoyelq/Documents/dfobot_data/GT_metadata.csv")
    gt_data = pd.read_csv(gt_file)
    return gt_data

def get_dataloaders():
    IMAGE_FOLDER_DIR = "/home/stoyelq/Documents/dfobot_data/image_folder/"
    BATCH_SZIE = 16
    NUM_WORKERS = 4

    data_transforms = {
        'train': transforms.Compose([
            transforms.ToTensor(),
            v2.ToDtype(torch.uint8, scale=True),
            v2.RandomRotation(180),
            v2.ToDtype(torch.float32, scale=True),
        ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
            v2.ToDtype(torch.float32, scale=True),
        ]),
    }

    data_dir = IMAGE_FOLDER_DIR
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
    dataloaders = {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=BATCH_SZIE, shuffle=True, num_workers=NUM_WORKERS)
        for x in ['train', 'val']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    return dataloaders, dataset_sizes, class_names




def train_model(model, criterion, optimizer, num_epochs=25, device="cuda"):
    since = time.time()
    dataloaders, dataset_sizes, class_names = get_dataloaders()
    # Create a temporary directory to save training checkpoints
    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')

        torch.save(model.state_dict(), best_model_params_path)
        best_acc = 0.0

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), best_model_params_path)

            print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

        # load best model weights
        model.load_state_dict(torch.load(best_model_params_path))
    return model


def imshow(inp, title=None):
    """Display image for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


def visualize_model(model, num_images=6, device="cuda"):
    dataloaders, dataset_sizes, class_names = get_dataloaders()
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(f'predicted: {class_names[preds[j]]}')
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)










#
# def import_data_old(num_train, num_test):
#     pt_list = os.listdir(AUGMENTED_DIR)
#     gt_df = load_dmapps_report()
#     data_dict = {"x_train": [], "y_train": [], "x_test": [], "y_test": [], "z_train": [], "z_test": []}
#
#     for pt_name in pt_list[:num_train]:
#         fish_id = pt_name.split("__")[0]
#         fish_data_row = gt_df[gt_df["specimen_identifier"] == fish_id]
#         fish_age = int(fish_data_row["annulus_count"].iloc[0])
#         fish_metadata = [int(fish_data_row["length_mm"].iloc[0]), fish_data_row["sex"].iloc[0],
#                          int(fish_data_row["weight_g"].iloc[0]), int(fish_data_row["collection_year"].iloc[0])]
#         image_tensor = torch.load(AUGMENTED_DIR + pt_name)
#         data_dict["x_train"].append(image_tensor)
#         data_dict["y_train"].append(fish_age)
#         data_dict["z_train"].append(fish_metadata)
#
#     for pt_name in pt_list[:num_test]:
#         fish_id = pt_name.split("__")[0]
#         fish_data_row = gt_df[gt_df["specimen_identifier"] == fish_id]
#         fish_age = int(fish_data_row["annulus_count"].iloc[0])
#         fish_metadata = [int(fish_data_row["length_mm"].iloc[0]), fish_data_row["sex"].iloc[0],
#                          int(fish_data_row["weight_g"].iloc[0]), int(fish_data_row["collection_year"].iloc[0])]
#         image_tensor = torch.load(AUGMENTED_DIR + pt_name)
#         data_dict["x_test"].append(image_tensor)
#         data_dict["y_test"].append(fish_age)
#         data_dict["z_test"].append(fish_metadata)
#
#     return data_dict

