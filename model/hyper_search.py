from solver import run_solver

dev_count = 1
device = f"cuda:{dev_count}"
learning_rates = [1e-6]
weight_decays = [0, 1e-7, 5e-7, 1e-6, 5e-6]
crop_sizes = [600]
config_dict = {
    "IMAGE_FOLDER_DIR": "/home/stoyelq/Documents/dfobot_data/image_folder/",
    # "IMAGE_FOLDER_DIR": "/home/stoyelq/Documents/dfobot_data/small_image_folder/",
    "NUM_WORKERS": 4,
    "VAL_CROP_SIZE": 600,
    "BATCH_SIZE": 20,
    "PRINT_EVERY": 100,
    "MAX_DATA": None, # 150,
    "NUM_EPOCHS": 5,
    "ACC_SAMPLES": 100,
    "ACC_VAL_SAMPLES": 100,
}

iter_count = 0
acc_history = []
for lr in learning_rates:
    for weight_decay in weight_decays:
        for cs in crop_sizes:
            iter_count += 1
            config_dict["CROP_SIZE"] = cs
            config_dict["LEARNING_RATE"] = lr
            config_dict["WEIGHT_DECAY"] = weight_decay
            solver = run_solver(device=device, plots=False,
                                all_layers=True, config_dict=config_dict, save_count=iter_count)
            acc_history.append(f"Learning rate: {lr}, crop size: {cs}, and weight_decay: {weight_decay}. \n Best validation accuracy: {solver.best_val_acc}")

for test_run in acc_history:
    print(test_run)
