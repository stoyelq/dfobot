
from model.solver import run_solver, make_solver_plots, make_bot_plot
import torch
import gc

def get_dfobot():
    solver_name = "model_solver_1_11.pt"
    solver = torch.load(f'/home/stoyelq/Documents/dfobot_data/{solver_name}')
    bot = solver.model
    bot.state_dict = solver.best_params
    return bot, solver

def get_next_image_prediction(solver, bot, device="cuda:0"):
    images, data, labels = next(iter(solver.val_dataloader))
    images = images.to(device)
    data = data.to(device)
    labels = labels.to(device)
    bot = bot.to(device)
    output = bot(images, data)
    return images, output, labels

bot, solver = get_dfobot()
imgs, outputs, labels = get_next_image_prediction(solver, bot)

import matplotlib.pyplot as plt
plt.imshow(imgs[0].cpu().permute(1, 2, 0))
plt.show()

y_pred, y_true = make_bot_plot(bot, 500, solver.config_dict, "cuda:0")
