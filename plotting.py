import matplotlib.pyplot as plt
import numpy as np

from params import *


def plot_gust_inputs(tsim, wgusts):
    fig = plt.figure(figsize=FIG_SIZE)
    ax1 = fig.add_subplot(111)

    (line1,) = ax1.plot(tsim, wgusts[:, 0], color=COLORS["primary"], linewidth=1, label="horizontal: $w_u$")
    ax1.set_ylim(-8, 8)
    ax1.set_ylabel("Horizontal gust velocity [ft/s]")
    ax1.tick_params(axis="y", labelcolor=COLORS["primary"])

    ax2 = ax1.twinx()
    (line2,) = ax2.plot(tsim, wgusts[:, 1], color=COLORS["secondary"], linewidth=1, label="vertical: $w_w$")
    ax2.set_ylim(-8, 8)
    ax2.set_ylabel("Vertical gust velocity [ft/s]")
    ax2.tick_params(axis="y", labelcolor=COLORS["secondary"])

    lines = [line1, line2]
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="best", fontsize=PLOT_STYLE["font.size"])

    ax1.set_xlabel("Time [s]")
    ax1.grid(True)
    plt.title("Turbulence, horizontal and vertical components")
    return fig


def plot_gust_response(tsim, gustresp):
    fig = plt.figure(figsize=FIG_SIZE)
    ax1 = fig.add_subplot(111)

    (line1,) = ax1.plot(tsim, gustresp[:, 0], color=COLORS["primary"], linewidth=1.5, label="$y_1$ velocity")
    ax1.set_ylim(-200, 200)
    ax1.set_ylabel("Horizontal velocity [ft/s]")
    ax1.tick_params(axis="y", labelcolor=COLORS["primary"])

    ax2 = ax1.twinx()
    (line2,) = ax2.plot(tsim, gustresp[:, 1], color=COLORS["secondary"], linewidth=1.5, label="$y_2$ climb rate")
    ax2.set_ylim(-200, 200)
    ax2.set_ylabel("Climb rate [ft/s]")
    ax2.tick_params(axis="y", labelcolor=COLORS["secondary"])

    lines = [line1, line2]
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="best", fontsize=PLOT_STYLE["font.size"])

    ax1.set_xlabel("Time [s]")
    ax1.grid(True)
    plt.title("Aircraft gust response (uncontrolled)")
    return fig


def plt_resp(t, x, labelstrs, colorstrs):
    """
    Plot mean and standard deviation of multiple trajectories

    Args:
        t: Time vector (1D array)
        x: 3D array of signals (channels, time_steps, samples)
        labelstrs: List of legend labels for each channel
        colorstrs: List of color specifications for each channel
    """
    # Validate input dimensions
    nu, nt, _ = x.shape
    t = t.flatten()
    if len(t) != nt:
        raise ValueError("Time vector must match x's time dimension")

    # Calculate statistics
    mean_vals = np.mean(x, axis=2)
    std_dev = np.std(x, axis=2)
    top_std = mean_vals + std_dev
    bot_std = mean_vals - std_dev

    # Get current axes
    ax = plt.gca()

    # First pass: plot means to establish legend
    handles = []
    for i in range(nu):
        (line,) = ax.plot(t, mean_vals[i], colorstrs[i], lw=1.5)
        handles.append(line)

    # Plot shaded std regions
    for i in range(nu):
        ax.fill_between(t, bot_std[i], top_std[i], color=colorstrs[i], alpha=0.15, linewidth=0)

    # Second pass: replot means on top
    for i in range(nu):
        ax.plot(t, mean_vals[i], colorstrs[i], lw=1.5)

    # Add legend and format
    ax.legend(handles, labelstrs, loc="best", fontsize=14)
    ax.grid(True)

    return ax
