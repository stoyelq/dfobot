import os
import pandas as pd
from PIL import Image
from torchvision import transforms
from torchvision.transforms import v2
import torch
import cv2

DATA_DIR = "/home/stoyelq/Documents/dfobot_data/preprocessing/"
CROP_DIR = "/home/stoyelq/Documents/dfobot_data/cropped_singles/"
TEST_DIR = "/home/stoyelq/Documents/dfobot_data/cropped_singles_test/"
AUGMENTED_DIR = "/home/stoyelq/Documents/dfobot_data/augmented/"
IMAGE_FOLDER_DIR = "/home/stoyelq/Documents/dfobot_data/image_folder/"

BUFFER_PX = 5
AREA_THRESHOLD = 0.5
OUT_DIM = (512, 512)
TEST_TRAIN_SPLIT = 0.90

def crop_and_save(img, contour, out_dir, buffer=5, outdim=(256, 256)):
    rect = cv2.boundingRect(contour)  # x, y, w, h
    x1 = rect[0] - buffer
    y1 = rect[1] - buffer
    x2 = (rect[0] + rect[2]) + buffer
    y2 = (rect[1] + rect[3]) + buffer
    cropped = img[y1:y2, x1:x2]
    try:
        scaled = cv2.resize(cropped, dsize=outdim)
        saved = cv2.imwrite(out_dir, scaled)
        if not saved:
            print("Could not save {out_dir}".format(out_dir=out_dir))
    except cv2.Error as e:
        print("Error {e}. Could not save {out_dir}".format(e=e, out_dir=out_dir))


def get_age_from_name(img_name, gt_df):
    fish_id = img_name.split("photo")[0][:-1].split(" ")[0]
    fish_data_row = gt_df[gt_df["specimen_identifier"] == fish_id]
    try:
        fish_age = int(fish_data_row["annulus_count"].iloc[0])
    except:
        return None, None
    return fish_age, fish_id

def crop_and_isolate():
    # load images
    img_list = os.listdir(DATA_DIR)
    count = len(img_list)
    gt_df = load_dmapps_report()

    for img_name in img_list:
        count += -1
        if count == 652:
            # that otolith touches edge of image and breaks things, ignore : )
            print(img_name)
            continue
        if count % 100 == 0:
            print(count)
        if count < TEST_TRAIN_SPLIT * len(img_list):
            mode = "train"
        else:
            mode = "val"

        fish_age, fish_id = get_age_from_name(img_name, gt_df)
        if fish_age is None:
            continue
        photo_count = int(img_name.split("photo")[1][1]) if "photo" in img_name else 1
        out_dir = f"{IMAGE_FOLDER_DIR}{mode}/{fish_age}/"

        img_path = DATA_DIR + img_name
        img = cv2.imread(img_path)
        if img is None:
            print(f"file {img_path} could not be read, check with os.path.exists()")
            continue
        # assert img is not None, f"file {img_path} could not be read, check with os.path.exists()"

        # clip on threshold and convert to grayscale:
        ret, thresh = cv2.threshold(img, 40, 255, 0)
        imgray = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)

        # Find contours and sort using contour area
        cnts = cv2.findContours(imgray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

        # always save largest otolith/contour:

        file_count = 2 * photo_count - 1
        crop_and_save(img, cnts[0], out_dir=f"{out_dir}{fish_id}__{file_count}.jpg", buffer=BUFFER_PX, outdim=OUT_DIM)
        cv2.drawContours(img, [cnts[0]], -1, (36, 255, 12), 3)

        # grab second otolith if area is closish:
        first_area = cv2.contourArea(cnts[0])
        second_area = cv2.contourArea(cnts[1])
        if second_area > AREA_THRESHOLD * first_area:
            file_count = 2 * photo_count
            crop_and_save(img, cnts[1], out_dir=f"{out_dir}{fish_id}__{file_count}.jpg", buffer=BUFFER_PX, outdim=OUT_DIM)


        # visualization tool:
        # cv2.drawContours(img, [cnts[1]], -1, (36,255,12), 3)
        # cv2.imshow('image', img)
        # cv2.waitKey()
        # cv2.destroyAllWindows()


def process_image(img):
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        v2.ToDtype(torch.uint8, scale=True),
        v2.Grayscale(1),
        v2.ToDtype(torch.float32, scale=True),
    ])

    img_out = preprocess(img)
    return img_out

def process_and_augment_image(img):
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        v2.ToDtype(torch.uint8, scale=True),
        v2.RandomRotation(180),
        v2.Grayscale(1),
        v2.ToDtype(torch.float32, scale=True),
    ])

    img_out = preprocess(img)
    return img_out


def load_dmapps_report():
    gt_file = os.path.join("/home/stoyelq/Documents/dfobot_data/GT_metadata.csv")
    gt_df = pd.read_csv(gt_file)
    return gt_df


def process_and_augment_images():
    for mode in ["train", "val"]:
        if mode == "train":
            img_dir = CROP_DIR
        else:
            img_dir = TEST_DIR
    img_list = os.listdir(img_dir)
    count = len(img_list)
    gt_df = load_dmapps_report()
    for img_name in img_list:
        count += -1
        if count % 100 == 0:
            print(count)
        fish_id = img_name.split("photo")[0][10:].split("_")[0][:-1].split(" ")[0]
        fish_data_row = gt_df[gt_df["specimen_identifier"] == fish_id]
        fish_age = int(fish_data_row["annulus_count"].iloc[0])

        # make file names unique based on photo, otolith number
        oto_count = int(img_name.split("otolith")[1][1]) if "otolith" in img_name else 1
        photo_count = int(img_name.split("photo")[1][1]) if "photo" in img_name else 1
        fish_count = oto_count + 2 * photo_count - 2
        out_dir = f"{AUGMENTED_DIR}{mode}/{fish_age}/"

        img_path = img_dir + img_name
        img = Image.open(img_path)
        image_tensor = process_image(img)
        torch.save(image_tensor, f"{out_dir}{fish_id}__{fish_count}.pt")
        augmented_img_1 = process_and_augment_image(img)
        augmented_img_2 = process_and_augment_image(img)
        torch.save(augmented_img_1, f"{out_dir}{fish_id}__{fish_count}_augmented_1.pt")
        torch.save(augmented_img_2, f"{out_dir}{fish_id}__{fish_count}_augmented_2.pt")

        # visulizers:
        # plt.imshow(augmented_img_1.permute(1, 2, 0))
        # plt.show()
        # plt.imshow(augmented_img_2.permute(1, 2, 0))
        # plt.show()
        # plt.imshow(image_tensor.permute(1, 2, 0))
        # plt.show()

