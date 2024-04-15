from solver import run_solver


config_dict = {
    "IMAGE_FOLDER_DIR": "/home/stoyelq/Documents/dfobot_data/image_folder/",
    # "IMAGE_FOLDER_DIR": "/home/stoyelq/Documents/dfobot_data/small_image_folder/",
    # "IMAGE_FOLDER_DIR": "/home/stoyelq/Documents/dfobot_data/ant_bees/",
    "NUM_WORKERS": 4,
    "CROP_SIZE": 600,
    "VAL_CROP_SIZE": 600,
    "LEARNING_RATE": 5e-6,
    "BATCH_SIZE": 20,
    "PRINT_EVERY": 3,
    "MAX_DATA": None, #250,
    "NUM_EPOCHS": 2,
    "WEIGHT_DECAY": 0,
    "ACC_SAMPLES": 30,
    "ACC_VAL_SAMPLES": 30,
}

run_solver(device="cuda:1",  plots=False, all_layers=False, save_count=8, config_dict=config_dict)
           # load_checkpoint="/home/stoyelq/Documents/dfobot_data/gpu_1/epoch_2.pkl")

