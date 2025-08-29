# Keep CUDA autotuner fast on fixed input sizes
import torch
torch.backends.cudnn.benchmark = True
