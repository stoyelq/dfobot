import os
import pandas as pd
from PIL import Image
from torchvision import transforms
from torchvision.transforms import v2
import torch
import cv2

# DATA_DIR = "/home/stoyelq/Documents/dfobot_data/preprocessing/"
DATA_DIR = "/home/stoyelq/Documents/dfobot_data/herring/enhanced/"
# CROP_DIR = "/home/stoyelq/Documents/dfobot_data/cropped_singles/"
CROP_DIR = "/home/stoyelq/Documents/dfobot_data/cropped_herring_singles/"
TEST_DIR = "/home/stoyelq/Documents/dfobot_data/cropped_singles_test/"
IMAGE_FOLDER_DIR = "/home/stoyelq/Documents/dfobot_data/image_folder/"

BUFFER_PX = 5
AREA_THRESHOLD = 0.5
OUT_DIM = (1000, 1000)
TEST_TRAIN_SPLIT = 0.90

def crop_and_save(img, contour, out_dir, buffer=5, outdim=(256, 256)):
    rect = cv2.boundingRect(contour)  # x, y, w, h
    x1 = max(rect[0] - buffer, 0)
    y1 = max(rect[1] - buffer, 0)
    x2 = min((rect[0] + rect[2]) + buffer, img.shape[1])
    y2 = min((rect[1] + rect[3]) + buffer, img.shape[0])
    cropped = img[y1:y2, x1:x2]
    try:
        scaled = cv2.resize(cropped, dsize=outdim)
        saved = cv2.imwrite(out_dir, scaled)
        if not saved:
            print("Could not save {out_dir}".format(out_dir=out_dir))
    except cv2.Error as e:
        print("Error {e}. Could not save {out_dir}".format(e=e, out_dir=out_dir))


def get_age_from_name(img_name, gt_df, herring=True):
    if herring:
        sample_id = img_name.split("-")[1]
        fish_number = img_name.split("-")[2].split(".")[0]
        try:
            fish_number = int(fish_number)
        except ValueError:
            fish_number = fish_number
        fish_id = img_name.split(".")[0][5:]
        fish_data_row = gt_df[(gt_df["sample_id"] == int(sample_id)) & (gt_df["fish_number"] == fish_number)]
    else:
        fish_id = img_name.split("photo")[0][:-1].split(" ")[0]
        fish_data_row = gt_df[gt_df["specimen_identifier"] == fish_id]
    try:
        fish_age = int(fish_data_row["annulus_count"].iloc[0])
        if fish_age == -99:
            return None, None
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
        #cv2.drawContours(img, [cnts[0]], -1, (36, 255, 12), 3)

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


def load_dmapps_report(herring=True):
    if herring:
        gt_file = os.path.join("/home/stoyelq/Documents/dfobot_data/2019_herring_GT.csv")
    else:
        gt_file = os.path.join("/home/stoyelq/Documents/dfobot_data/GT_metadata.csv")
    gt_df = pd.read_csv(gt_file)
    return gt_df

crop_and_isolate()