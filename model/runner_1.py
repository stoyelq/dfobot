from solver import run_solver


config_dict = {
    "IMAGE_FOLDER_DIR": "/home/stoyelq/Documents/dfobot_data/meta_image_folder/",
    "NUM_WORKERS": 4,
    "CROP_SIZE": 600,
    "VAL_CROP_SIZE": 600,
    "LEARNING_RATE": 1e-6,
    "BATCH_SIZE": 20,
    "PRINT_EVERY": 25,
    "MAX_DATA": None, # 150
    "NUM_EPOCHS": 2,
    "WEIGHT_DECAY": 1e-7,
    "ACC_SAMPLES": 1000,
    "ACC_VAL_SAMPLES": 1000,
}

run_solver(device="cuda:1",  plots=False, all_layers=True, save_count=13, config_dict=config_dict)
           # load_checkpoint="/home/stoyelq/Documents/dfobot_data/gpu_1/epoch_1.pkl")

