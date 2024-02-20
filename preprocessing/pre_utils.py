import os
import pandas as pd
from PIL import Image
from torchvision import transforms
from torchvision.transforms import v2
import torch
import cv2

DATA_DIR = "/home/stoyelq/Documents/dfobot_data/preprocessing/"

def load_dmapps_report():
    gt_file = os.path.join("/home/stoyelq/Documents/dfobot_data/GT_metadata.csv")
    gt_data = pd.read_csv(gt_file)
    return gt_data
def img_paths():
    return os.listdir(DATA_DIR)

def split_image(img_path):
    import cv2
    import numpy as np
    import os
    # Read image
    DATA_DIR = "/home/stoyelq/Documents/dfobot_data/preprocessing/"
    img_list = os.listdir(DATA_DIR)
    img_path = DATA_DIR + img_list[0]

    img = cv2.imread(img_path)
    assert img is not None, "file could not be read, check with os.path.exists()"
    ret, thresh = cv2.threshold(img, 40, 255, 0)
    imgray = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
    # Find contours and sort using contour area

    cnts = cv2.findContours(imgray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    for c in cnts[:2]:
        # Highlight largest contour
        cv2.drawContours(img, [c], -1, (36, 255, 12), 3)

    cv2.imshow('thresh', thresh)
    cv2.imshow('image', img)
    cv2.waitKey()
    cv2.destroyAllWindows()


def process_image(img_path):
    img = Image.open(img_path)

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        v2.ToDtype(torch.uint8, scale=True),
        v2.Grayscale(1),
        transforms.Resize(512),
        transforms.CenterCrop(488),
        v2.ToDtype(torch.float32, scale=True),
        transforms.Normalize(
            mean=[0.406],
            std=[0.225]
        )
    ])

    img_out = preprocess(img)
    print(img_out.shape)
    return img_out

def process_and_augment_image(processed_img):
    img = processed_img
    preprocess = transforms.Compose([
        v2.RandomRotation(180),
    ])

    img_out = preprocess(img)
    print(img_out.shape)
    return img_out
