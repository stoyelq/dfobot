
from solver import run_solver, make_solver_plots
import torch
solver_num = 0
solver = torch.load(f'/home//stoyelq/Documents/dfobot_data/hyper_search/model_solver_1.pt')
make_solver_plots(solver)