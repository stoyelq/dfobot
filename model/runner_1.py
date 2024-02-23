from solver import run_solver

config_dict = {
    # "IMAGE_FOLDER_DIR": "/home/stoyelq/Documents/dfobot_data/image_folder/",
    "IMAGE_FOLDER_DIR": "/home/stoyelq/Documents/dfobot_data/small_image_folder/",
    # "IMAGE_FOLDER_DIR": "/home/stoyelq/Documents/dfobot_data/ant_bees/",
    "NUM_WORKERS": 4,
    "CROP_SIZE": 600,
    "VAL_CROP_SIZE": 800,
    "LEARNING_RATE": 0.01,
    "BATCH_SIZE": 100,
    "PRINT_EVERY": 5,
    "MAX_DATA": None, # 150,
    "NUM_EPOCHS": 200,
    "WEIGHT_DECAY": 0, #1e-4,
    "ACC_SAMPLES": 100,
    "ACC_VAL_SAMPLES": 100,
}

run_solver(device="cuda:1",  plots=True, all_layers=False, config_dict=config_dict)

