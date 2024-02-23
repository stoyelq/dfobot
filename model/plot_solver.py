
from solver import run_solver, make_solver_plots
import torch
solver_num = 0
solver = torch.load(f'/home//stoyelq/Documents/dfobot_data/model_solver_{solver_num}.pt')
make_solver_plots(solver)