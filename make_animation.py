import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


ARRAY = [0.5, 0.2, 0.7, 0.4, 0.1, 0.6]
LOG_PATH = Path(__file__).parent / "animation_log.csv"
GIF_PATH = Path(__file__).parent / "tausort_animation.gif"

PALETTE = ["#FF006E", "#3A86FF", "#FFBE0B", "#8338EC", "#FB5607", "#00B4D8"]
LOSS_COLOR = "#FFFFFF"
MARKER_COLOR = "#FF006E"
BACKGROUND = "#0A0A0A"
SPINE_COLOR = "#444444"

NUM_TRAINING_FRAMES = 80
HOLD_FRAMES = 12
FPS = 6
LOSS_FLOOR = 1e-8
LOSS_THRESHOLD = 1e-8


def load_log():
    epochs = []
    losses = []
    indices_history = []
    perm_history = []
    with open(LOG_PATH) as csv_file:
        reader = csv.reader(csv_file)
        next(reader)
        array_length = len(ARRAY)
        for row in reader:
            epochs.append(int(row[0]))
            losses.append(max(float(row[1]), LOSS_FLOOR))
            indices_history.append([float(v) for v in row[2:2 + array_length]])
            perm_history.append([int(v) for v in row[2 + array_length:2 + 2 * array_length]])
    return epochs, losses, indices_history, perm_history


def pick_frame_indices(total_epochs):
    if total_epochs <= NUM_TRAINING_FRAMES:
        training_indices = list(range(total_epochs))
    else:
        sampled = np.linspace(0, total_epochs - 1, NUM_TRAINING_FRAMES).astype(int)
        training_indices = sampled.tolist()
    last_index = total_epochs - 1
    hold_tail = [last_index] * HOLD_FRAMES
    return training_indices + hold_tail


def style_value_bar_axis(ax, title):
    ax.set_title(title, fontsize=20, fontweight="bold", color="white", pad=10)
    ax.set_xlim(-0.5, len(ARRAY) - 0.5)
    ax.set_ylim(0, 1.15)
    ax.set_xticks([])
    ax.set_yticks(np.arange(0, 1.05, 0.1))
    ax.set_yticklabels([])
    ax.tick_params(axis="y", colors=SPINE_COLOR, length=5, width=1)
    for spine in ax.spines.values():
        spine.set_color(SPINE_COLOR)


def style_loss_axis(ax, total_epochs, min_loss, max_loss):
    ax.set_title("Loss (log10)", fontsize=20, fontweight="bold", color="white", pad=10)
    ax.set_xlim(0, total_epochs * 1.1)
    ax.set_yscale("log")
    ax.set_ylim(min_loss / 3, max_loss * 2)
    ax.set_xticks([])
    ax.tick_params(axis="y", colors="white", labelsize=10)
    for spine in ax.spines.values():
        spine.set_color(SPINE_COLOR)


def sorted_values_at(epoch_index, perm_history):
    permutation = perm_history[epoch_index]
    return [ARRAY[item_id] for item_id in permutation]


def sorted_colors_at(epoch_index, perm_history):
    permutation = perm_history[epoch_index]
    return [PALETTE[item_id] for item_id in permutation]


def main():
    plt.style.use("dark_background")
    epochs, losses, indices_history, perm_history = load_log()
    total_epochs = len(epochs)
    frame_indices = pick_frame_indices(total_epochs)

    figure, axes = plt.subplots(2, 2, figsize=(8, 8))
    figure.patch.set_facecolor(BACKGROUND)

    array_axis = axes[0, 0]
    indices_axis = axes[0, 1]
    sorted_axis = axes[1, 0]
    loss_axis = axes[1, 1]

    item_positions = list(range(len(ARRAY)))

    style_value_bar_axis(array_axis, "Array")
    array_axis.bar(
        item_positions,
        ARRAY,
        color=PALETTE,
        edgecolor="white",
        linewidth=1.5,
        width=0.6,
        zorder=3,
    )
    for x, value in zip(item_positions, ARRAY):
        array_axis.text(
            x,
            value + 0.04,
            f"{value:.1f}",
            ha="center",
            color="white",
            fontsize=13,
            fontweight="bold",
        )

    style_value_bar_axis(indices_axis, "Indices")
    indices_bars = indices_axis.bar(
        item_positions,
        indices_history[0],
        color=PALETTE,
        edgecolor="white",
        linewidth=1.5,
        width=0.6,
        zorder=3,
    )

    style_value_bar_axis(sorted_axis, "Sorted")
    initial_sorted_values = sorted_values_at(0, perm_history)
    initial_sorted_colors = sorted_colors_at(0, perm_history)
    sorted_bars = sorted_axis.bar(
        item_positions,
        initial_sorted_values,
        color=initial_sorted_colors,
        edgecolor="white",
        linewidth=1.5,
        width=0.6,
        zorder=3,
    )
    sorted_value_texts = []
    for x, value in zip(item_positions, initial_sorted_values):
        text = sorted_axis.text(
            x,
            value + 0.04,
            f"{value:.1f}",
            ha="center",
            color="white",
            fontsize=13,
            fontweight="bold",
        )
        sorted_value_texts.append(text)

    style_loss_axis(loss_axis, total_epochs, min(losses), max(losses))
    loss_line, = loss_axis.plot([], [], color=LOSS_COLOR, linewidth=3)
    loss_axis.axhline(
        y=LOSS_THRESHOLD,
        color=MARKER_COLOR,
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
        zorder=2,
    )
    loss_axis.text(
        total_epochs * 0.55,
        LOSS_THRESHOLD * 5,
        "stop: loss < 1e-8",
        color=MARKER_COLOR,
        fontsize=12,
        fontweight="bold",
        ha="left",
    )

    epoch_text = figure.text(
        0.5,
        0.965,
        "",
        ha="center",
        color=MARKER_COLOR,
        fontsize=26,
        fontweight="bold",
    )
    figure.tight_layout(rect=[0, 0, 1, 0.93])

    training_count = len(frame_indices) - HOLD_FRAMES

    def update(frame_number):
        epoch_index = frame_indices[frame_number]
        if frame_number >= training_count:
            epoch_text.set_text("Sorted!")
        else:
            epoch_text.set_text(f"epoch {epochs[epoch_index]}")
        for bar, indices_value in zip(indices_bars, indices_history[epoch_index]):
            bar.set_height(indices_value)
        current_sorted_values = sorted_values_at(epoch_index, perm_history)
        current_sorted_colors = sorted_colors_at(epoch_index, perm_history)
        for bar, value, color, text in zip(sorted_bars, current_sorted_values, current_sorted_colors, sorted_value_texts):
            bar.set_height(value)
            bar.set_facecolor(color)
            text.set_position((text.get_position()[0], value + 0.04))
            text.set_text(f"{value:.1f}")
        loss_line.set_data(epochs[:epoch_index + 1], losses[:epoch_index + 1])
        return (loss_line, epoch_text, *indices_bars, *sorted_bars, *sorted_value_texts)

    animation = FuncAnimation(
        figure,
        update,
        frames=len(frame_indices),
        interval=1000 // FPS,
        blit=False,
    )

    animation.save(GIF_PATH, writer=PillowWriter(fps=FPS))
    print(f"gif written to {GIF_PATH}")


if __name__ == "__main__":
    main()
