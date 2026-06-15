#!/usr/bin/env python3
"""
Keyboard Layout Optimization via Simulated Annealing


Notes:
- Cost is total Euclidean distance between consecutive characters.
- Coordinates are fixed (QWERTY-staggered grid). Optimization swaps assignments.

This base code uses Python "types" - these are optional, but very helpful
for debugging and to help with editing.

"""

import argparse
import json
import math
import os
import random
import string
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt  # type: ignore


Point = Tuple[float, float]
Layout = Dict[str, Point]


def qwerty_coordinates(chars: str) -> Layout:
    """Return QWERTY grid coordinates for the provided character set.

    The grid is a simple staggered layout (units are arbitrary):
    - Row 0: qwertyuiop at y=0, x in [0..9]
    - Row 1: asdfghjkl at y=1, x in [0.5..8.5]
    - Row 2: zxcvbnm at y=2, x in [1..6]
    - Space at (4.5, 3)
    Characters not present in the grid default to the space position.
    """
    row0 = "qwertyuiop"
    row1 = "asdfghjkl"
    row2 = "zxcvbnm"

    coords: Layout = {}
    for i, c in enumerate(row0):
        coords[c] = (float(i), 0.0)
    for i, c in enumerate(row1):
        coords[c] = (0.5 + float(i), 1.0)
    for i, c in enumerate(row2):
        coords[c] = (1.0 + float(i), 2.0)
    coords[" "] = (4.5, 3.0)

    # Backfill for requested chars; unknowns get space position.
    space_xy = coords[" "]
    for ch in chars:
        if ch not in coords:
            coords[ch] = space_xy
    return coords


def initial_layout() -> Layout:
    """Create an initial layout mapping chars to some arbitrary positions of letters."""

    # Start with identity for letters and space; others mapped to space.
    base_keys = "abcdefghijklmnopqrstuvwxyz "

    # Get coords - or use coords of space as default
    layout = qwerty_coordinates(base_keys)
    return layout


def preprocess_text(text: str, chars: str) -> str:
    """Lowercase and filter to the allowed character set; map others to space."""
    text = text.lower()  # Converts to lowercases
    result = ""  # Result to store the processed text
    for char in text:
        if char not in chars:
            result += " "  # Replace other than alphabets and spaces as spaces
        else:
            result += char  # Adding alphabets and spaces to result
    return result


def path_length_cost(text: str, layout: Layout) -> float:
    """Sum Euclidean distances across consecutive characters in text."""

    dist = 0
    for i in range(len(text) - 1):
        x1, y1 = layout[text[i]]  # Tuple coordinates of the first point
        x2, y2 = layout[text[i + 1]]  # Tuple coordinates of second point
        dist += ((y2 - y1) ** 2 + (x2 - x1) ** 2) ** 0.5  # Calculating distance
    return dist


######
# Define any other useful functions, such as to create new layout etc.


def random_swap(layout: Layout) -> Layout:
    """Takes layout and gives new layout after swaping."""

    new_layout = dict(layout)  # Independent copy

    keys = "abcdefghijklmnopqrstuvwxyz "
    char1, char2 = random.sample(
        keys, 2
    )  # Selecting two random keys without replacement
    new_layout[char1], new_layout[char2] = (
        new_layout[char2],
        new_layout[char1],
    )  # Swaping the coordinates of the selected keys
    return new_layout


######


# Dataclass is like a C struct - you can use it just to store data if you wish
# It provides some convenience functions for assignments, printing etc.
@dataclass
class SAParams:
    iters: int = 50000
    t0: float = 1.0  # Initial temperature setting
    alpha: float = 0.999  # geometric decay per iteration
    epoch: int = 1  # iterations per temperature step (1 = per-iter decay)


def simulated_annealing(
    text: str,
    layout: Layout,
    params: SAParams,
    rng: random.Random,
) -> Tuple[Layout, float, List[float], List[float]]:
    """Simulated annealing to minimize path-length cost over character swaps.

    Returns best layout, best cost, and two lists:
    - best cost up to now (monotonically decreasing)
    - cost of current solution (may occasionally go up)
    These will be used for plotting
    """

    # Assigning variables from the class params: SAParams
    T = params.t0
    alpha = params.alpha
    iters = params.iters
    epoch = params.epoch

    # Initialising variables to trace best costs and current costs
    best_trace = []
    current_trace = []

    text = preprocess_text(
        text, "abcdefghijklmnopqrstuvwxyz "
    )  # Using preprocess function to modify the text

    current_cost, best_cost = path_length_cost(text, layout), path_length_cost(
        text, layout
    )  # Using Euclidean distance to calculate cost

    current_layout = dict(
        layout
    )  # A copy of layout is assigned to initial current_layout
    best_layout = dict(layout)  # A copy of layout is assigned to initial best_layout

    for i in range(iters):

        candidate_layout = random_swap(
            current_layout
        )  # Swaping the layout and checking
        candidate_cost = path_length_cost(text, candidate_layout)  # Calculating cost

        delta = (
            candidate_cost - current_cost
        )  # detla difference betweeen current_cost and candidate_cost

        # If delta negitive(slope negitive) we change the current parameters and current parameters can changes with probability exp(-delta/T)
        if delta < 0 or random.random() < math.exp(-delta / (T)):
            current_cost = candidate_cost
            current_layout = dict(candidate_layout)
        # If current_cost is less than best_cost changing the parameters
        if best_cost > current_cost:
            best_cost = current_cost
            best_layout = dict(current_layout)

        if (
            i % epoch == 0
        ):  # Changing Temperature(in this case every iteration because epoch=1)
            T = T * alpha

        # Appending the best_cost and current_cost
        best_trace.append(best_cost)
        current_trace.append(current_cost)

    return (best_layout, best_cost, best_trace, current_trace)


def plot_costs(
    layout: Layout, best_trace: List[float], current_trace: List[float]
) -> None:

    # Plot cost trace
    out_dir = "."
    plt.figure(figsize=(6, 3))
    plt.plot(best_trace, lw=1.5)
    plt.plot(current_trace, lw=1.5)
    plt.xlabel("Iteration")
    plt.ylabel("Best Cost")
    plt.title("Best Cost vs Iteration")
    plt.tight_layout()
    path = os.path.join(out_dir, f"cost_trace.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")

    # Plot layout scatter
    xs, ys, labels = [], [], []
    for ch, (x, y) in layout.items():
        xs.append(x)
        ys.append(y)
        labels.append(ch)

    plt.figure(figsize=(6, 3))
    plt.scatter(xs, ys, s=250, c="#1f77b4")
    for x, y, ch in zip(xs, ys, labels):
        plt.text(
            x,
            y,
            ch,
            ha="center",
            va="center",
            color="white",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.15", fc="#1f77b4", ec="none", alpha=0.9),
        )
    plt.gca().invert_yaxis()
    plt.title("Optimized Layout")
    plt.axis("equal")
    plt.tight_layout()
    path = os.path.join(out_dir, f"layout.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def load_text(filename) -> str:
    if filename is not None:
        with open(filename, "r", encoding="utf-8") as f:
            return f.read()
    # Fallback demo text
    return (
        "the quick brown fox jumps over the lazy dog\n" "APL is the best course ever\n"
    )


def main(filename: str | None = None) -> None:
    rng = random.Random(0)
    chars = "abcdefghijklmnopqrstuvwxyz "

    # Initial assignment - QWERTY
    layout0 = initial_layout()

    # Prepare text and evaluate baseline
    raw_text = load_text(filename)

    text = preprocess_text(raw_text, chars)

    baseline_cost = path_length_cost(text, layout0)
    print(f"Baseline (QWERTY assignment) cost: {baseline_cost:.4f}")

    # Annealing - give parameter values
    params = SAParams()
    start = time.time()
    best_layout, best_cost, best_trace, current_trace = simulated_annealing(
        text, layout0, params, rng
    )
    dur = time.time() - start
    print(
        f"Optimized cost: {best_cost:.4f}  (improvement {(baseline_cost - best_cost):.4f})"
    )
    print(f"Runtime: {dur:.2f}s over {params.iters} iterations")

    plot_costs(best_layout, best_trace, current_trace)

    # plot_costs(layout0, [], [])


if __name__ == "__main__":
    main()
