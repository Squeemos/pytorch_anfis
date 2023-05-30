import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.pyplot import cm

import torch
from torch import nn
from torch.nn import functional as F
from torch import optim

from pathlib import Path

from anfis.anfis import AnfisLayer

def dccn(tensor):
    return tensor.detach().clone().cpu().numpy()

def simple_loss(val1, val2, loss_fn):
    return loss_fn(
        torch.tensor(val1, dtype=torch.float32),
        torch.tensor(val2, dtype=torch.float32),
    ).item()

def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_fn = nn.MSELoss()
    domain = 6
    eps = .5
    function = lambda x: (torch.sin(2 * x) * x) / 3
    fn = lambda x: (np.sin(2 * x) * x) / 3
    testing_size = 1_000
    x = torch.linspace(-domain, domain, testing_size).to(device).reshape(-1, 1)
    range_x = dccn(function(x))
    max_y = range_x.max()
    min_y = range_x.min()

    n_rules = 8

    path = Path("model.pt")
    model = nn.Sequential(
        nn.Linear(1, 10),
        nn.Linear(10, 1),
        AnfisLayer(1, n_rules=8),
    ).to(device)
    model.load_state_dict(torch.load(path))
    model.eval()
    model_out = dccn(model(x)).flatten()
    colors = iter(cm.bwr(np.linspace(0, 1, n_rules)))

    x = dccn(x).flatten()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 10))
    plt.subplots_adjust(bottom=.1)
    ax1.set_xlim(-domain - eps, domain + eps)
    ax1.set_ylim(min_y - eps, max_y + eps)

    centers = model[-1].centers.clone().detach().cpu().numpy()
    widths = model[-1].widths.clone().detach().cpu().numpy()

    min_idx = np.unravel_index(np.argmin(centers), centers.shape)
    max_idx = np.unravel_index(np.argmax(centers), centers.shape)

    line_plots = []
    middle_x = np.linspace(-7.5, 7.5, testing_size)
    for idx, (c, w) in enumerate(zip(centers, widths)):
        for idx_minor, (center, width) in enumerate(zip(c, w)):
            middle_y = np.exp(-((middle_x - center)**2 / (2 * width**2)))
            rule_plot, = ax2.plot(middle_x, middle_y, label=f"output_rule: {idx}" if idx_minor == 0 else None)
            line_plots.append(rule_plot)

    initial_x = 0.0
    initial_y = fn(initial_x)
    initial_y_model = dccn(
        model(torch.tensor(
            [initial_x],
        ).to(device).reshape(-1, 1))
    ).item()

    ax1.set_title(f"Point Loss: {simple_loss(initial_y, initial_y_model, loss_fn):.2f}")

    point_on, = ax1.plot(initial_x, initial_y, "bo", markersize=8)
    points_off = []
    for val in range(n_rules):
        pt, = ax2.plot(0, 0, "go", markersize=8)
        points_off.append(pt)
    point_off, = ax1.plot(0, 0, "ro", markersize=8)
    slider_ax = plt.axes([.2, .01, .6, .03])
    slider = Slider(slider_ax, "x", -domain, domain, valinit=initial_x, valstep=x)

    ax1.plot(x, fn(x))
    ax1.plot(x, model_out)

    def update(val):
        x = slider.val
        y = fn(val)
        point_on.set_xdata([x])
        point_on.set_ydata([y])

        model_y = dccn(
            model(torch.tensor(
                [x],
            ).to(device).reshape(-1, 1))
        ).item()
        point_off.set_xdata([x])
        point_off.set_ydata([model_y])

        before_rule = dccn(
            model[:-1](torch.tensor(
                [x],
            ).to(device).reshape(-1, 1))
        ).item()

        for idx, (c, w) in enumerate(zip(centers, widths)):
            for idx_minor, (center, width) in enumerate(zip(c, w)):
                rule_y = np.exp(-((before_rule - center)**2 / (2 * width**2)))
                points_off[idx_minor].set_xdata([before_rule])
                points_off[idx_minor].set_ydata([rule_y])
                line_plots[idx_minor].set_alpha(max(.1, rule_y))


        ax1.set_title(f"Point Loss: {simple_loss(y, model_y, loss_fn):.2f}")
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
