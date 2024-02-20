import os
import pandas as pd

def load_dmapps_report():
    gt_file = os.path.join("/home/stoyelq/Documents/dfobot_data/GT_metadata.csv")
    gt_data = pd.read_csv(gt_file)
    return gt_data


