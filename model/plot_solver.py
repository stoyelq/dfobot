
from solver import run_solver, make_solver_plots, make_bot_plot
import torch

# solver_name = "hyper_search/model_solver_1_7.pt"
solver_name = "model_solver_0.pt"
solver = torch.load(f'/home/stoyelq/Documents/dfobot_data/{solver_name}')

bot_name = "dfobot.pt"
bot = torch.load(f'/home/stoyelq/Documents/dfobot_data/{bot_name}')
device = "cuda:0"
bot.to(device)
# save dfobot:

if False:
    best_model = solver.model
    best_model.state_dict = solver.best_params
    torch.save(best_model, f"/home/stoyelq/Documents/dfobot_data/dfobot.pt")

if False:
    imgs, labels = next(iter(solver.val_dataloader))
    imgs = imgs.to("cuda:0")
    output = bot(imgs)

    imgs, labels = next(iter(solver.train_dataloader))
    imgs = imgs.to("cuda:0")
    output = bot(imgs)
#
# print(solver.best_val_acc)
# make_solver_plots(solver)
make_bot_plot(bot, 70, solver.config_dict, device)
