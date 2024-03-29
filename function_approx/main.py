import torch
from torch import nn
from torch import optim
from matplotlib import pyplot as plt
from pathlib import Path
from anfis.anfis import AnfisLayer
from time import perf_counter
from tqdm import tqdm


@torch.jit.script
def function(x):
    return (torch.sin(2 * x) * x) / 3


def dccn(tensor):
    return tensor.detach().clone().cpu().numpy()


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.Sequential(
        nn.Linear(1, 20),
        nn.Linear(20, 1),
        AnfisLayer(1, n_rules=16, normalize_rules=False, membership_type="Gaussian"),  # noqa: E501
    ).to(device)

    loss_fn = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=.01)

    training_its = 12_000
    training_size = 1_000
    testing_size = 1_000
    domain = 8

    start = perf_counter()

    for it in tqdm(range(training_its)):
        x = (0.5 - torch.rand((training_size, 1), device=device)) * 2 * domain
        y = function(x)
        output = model(x)

        loss = loss_fn(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    end = perf_counter()

    print(f"Training complete. Took {end - start:.2f} seconds.")
    path = Path("model.pt")
    torch.save(model.state_dict(), path)

    fig = plt.figure(figsize=(10, 10))   # noqa: F841
    model.eval()

    with torch.no_grad():
        x = torch.linspace(-domain, domain, testing_size).to(device).reshape(-1, 1)  # noqa: E501
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
    main()
