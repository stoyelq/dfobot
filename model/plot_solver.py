
from solver import run_solver, make_solver_plots
import torch

solver_name = "hyper_search/model_solver_1_1_old.pt"
solver_name = "model_solver_0.pt"
solver = torch.load(f'/home/stoyelq/Documents/dfobot_data/{solver_name}')

bot_name = "dfobot.pt"
bot = torch.load(f'/home/stoyelq/Documents/dfobot_data/{bot_name}')

# save dfobot:

if True:
    best_model = solver.model
    best_model.state_dict = solver.best_params
    torch.save(best_model, f"/home/stoyelq/Documents/dfobot_data/dfobot.pt")

if True:
    imgs, labels = next(iter(solver.val_dataloader))
    imgs = imgs.to("cuda:0")
    output = bot(imgs)

    imgs, labels = next(iter(solver.train_dataloader))
    imgs = imgs.to("cuda:0")
    output = bot(imgs)
#
# print(solver.best_val_acc)
# make_solver_plots(solver)
