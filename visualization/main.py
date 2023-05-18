import torch
from torch import nn
from torch.nn import functional as F
from torch import optim

from anfis.anfis import AnfisLayer

def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.Sequential(
        nn.Linear(1, 1),
        nn.ReLU(),
        AnfisLayer(1, n_rules=2),
    ).to(device)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
