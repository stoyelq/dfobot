from solver import run_solver


config_dict = {
    "IMAGE_FOLDER_DIR": "/home/stoyelq/Documents/dfobot_data/image_folder/",
    "NUM_WORKERS": 4,
    "CROP_SIZE": 600,
    "VAL_CROP_SIZE": 600,
    "LEARNING_RATE": 1e-5,
    "BATCH_SIZE": 20,
    "PRINT_EVERY": 25,
    "MAX_DATA": None, #250,
    "NUM_EPOCHS": 20,
    "WEIGHT_DECAY": 0,
    "ACC_SAMPLES": 50,
    "ACC_VAL_SAMPLES": 50,
}

run_solver(device="cuda:1",  plots=False, all_layers=False, save_count=8-1, config_dict=config_dict,
           load_checkpoint="/home/stoyelq/Documents/dfobot_data/gpu_1/run_8/epoch_20.pkl")

