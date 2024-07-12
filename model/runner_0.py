
from solver import run_solver

config_dict = {
    "IMAGE_FOLDER_DIR": "/home/stoyelq/Documents/dfobot_data/small_meta_image_folder/",
    "NUM_WORKERS": 4,
    "CROP_SIZE": 200,
    "VAL_CROP_SIZE": 200,
    "LEARNING_RATE": 1e-6,
    "BATCH_SIZE": 20,
    "PRINT_EVERY": 25,
    "MAX_DATA": None, # 150,
    "NUM_EPOCHS": 2,
    "WEIGHT_DECAY": 0,
    "ACC_SAMPLES": 100,
    "ACC_VAL_SAMPLES": 100,
}

run_solver(device="cuda:0", plots=False, all_layers=True, save_count='15-0', config_dict=config_dict)
           # load_checkpoint="/home/stoyelq/Documents/dfobot_data/gpu_0/run_7-1/epoch_25.pkl")

