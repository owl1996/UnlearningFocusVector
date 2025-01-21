import torch

if torch.cuda.is_available():
    torch.cuda.set_device(int(args.gpu))
    device = torch.device(f"cuda:{int(args.gpu)}")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

import torch
print(torch.__version__)
print(torch.backends.mps.is_available())  # Devrait retourner True
print(device)