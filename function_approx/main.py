import torch
from torch import nn
from torch.nn import functional as F
from torch import optim

from matplotlib import pyplot as plt
import numpy as np

from pathlib import Path

from anfis.anfis import AnfisLayer


def dccn(tensor):
    return tensor.detach().clone().cpu().numpy()

def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.Sequential(
        nn.Linear(1, 10),
        nn.Linear(10, 1),
        AnfisLayer(1, n_rules=8),
    ).to(device)

    function = lambda x: (torch.sin(2 * x) * x) / 3
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=.01)

    training_its = 4_000
    training_size = 100
    testing_size = 1_000
    domain = 8

    for it in range(training_its):
        x = (0.5 - torch.rand((training_size, 1), device=device)) * 2 * domain
        y = function(x)
        output = model(x)

        loss = loss_fn(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Loss: {loss.item():.5f}", end="\r")

    print("Training complete")
    path = Path("model.pt")
    torch.save(model.state_dict(), path)

    fig = plt.figure(figsize=(10, 10))
    model.eval()

    x = torch.linspace(-domain, domain, testing_size).to(device).reshape(-1, 1)
    y = function(x)
    model_y = model(x)

    x = dccn(x)
    y = dccn(y)
    model_y = dccn(model_y)

    plt.plot(x, y, label="real")
    plt.plot(x, model_y, label="anfis")
    plt.legend()
    plt.show()

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
