
from solver import run_solver

config_dict = {
    "IMAGE_FOLDER_DIR": "/home/stoyelq/Documents/dfobot_data/image_folder/",
    "NUM_WORKERS": 4,
    "CROP_SIZE": 600,
    "VAL_CROP_SIZE": 600,
    "LEARNING_RATE": 5e-7,
    "BATCH_SIZE": 20,
    "PRINT_EVERY": 25,
    "MAX_DATA": None, # 150,
    "NUM_EPOCHS": 10,
    "WEIGHT_DECAY": 0,
    "ACC_SAMPLES": 1000,
    "ACC_VAL_SAMPLES": 1000,
}

run_solver(device="cuda:0", plots=True, all_layers=True, save_count=7-2, config_dict=config_dict,
           load_checkpoint="/home/stoyelq/Documents/dfobot_data/gpu_0/run_7-1/epoch_25.pkl")

