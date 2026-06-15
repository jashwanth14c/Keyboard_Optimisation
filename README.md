# Keyboard Layout Optimization using Simulated Annealing

## Overview
This project uses Simulated Annealing to optimize keyboard layouts by minimizing the total finger travel distance required to type a given text. Starting from a QWERTY layout, character positions are swapped and evaluated using a Euclidean distance-based cost function.

## Features
- QWERTY keyboard coordinate model
- Text preprocessing
- Euclidean distance cost calculation
- Simulated Annealing optimization
- Cost and layout visualization


## Usage

Run with the default sample text:

```bash
python kbd_optimi.py
```

Run with a custom text file:

```bash
python kbd_optimi.py input.txt
```

## Output

- `cost_trace.png` – Optimization progress over iterations
- `layout.png` – Optimized keyboard layout visualization

## Algorithm

1. Start with the QWERTY layout.
2. Randomly swap two character positions.
3. Compute the new typing cost.
4. Accept better solutions or probabilistically accept worse ones.
5. Gradually reduce temperature until completion.

## Concepts Used

- Simulated Annealing
- Local Search
- Euclidean Distance
- Optimization Algorithms

