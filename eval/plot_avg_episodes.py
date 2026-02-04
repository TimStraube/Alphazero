#!/usr/bin/env python3
"""
Scan TensorBoard logs under `logs/` and plot episode-length / avg_episodes series.
Saves to a PNG file.
"""
import os
import argparse
import matplotlib.pyplot as plt

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except Exception:
    EventAccumulator = None


def load_series(run_dir):
    if EventAccumulator is None:
        raise RuntimeError("tensorboard is required to read event files (pip install tensorboard)")
    ea = EventAccumulator(run_dir, size_guidance={"scalars": 0})
    ea.Reload()
    tags = ea.Tags().get('scalars', [])
    series = {}
    for tag in tags:
        if 'avg_episodes' in tag or 'episode_length' in tag:
            events = ea.Scalars(tag)
            steps = [e.step for e in events]
            vals = [e.value for e in events]
            series[tag] = (steps, vals)
    return series


def smooth_values(vals, smooth_factor: float):
    """Apply exponential smoothing like TensorBoard: smooth in [0,1].
    Higher smooth_factor => smoother (0.9 means heavy smoothing).
    """
    if smooth_factor <= 0.0:
        return vals
    out = []
    last = vals[0]
    alpha = smooth_factor
    for v in vals:
        last = last * alpha + v * (1.0 - alpha)
        out.append(last)
    return out


def main(logdir, out, runs_arg=None, smooth=0.0):
    runs = sorted([d for d in os.listdir(logdir) if os.path.isdir(os.path.join(logdir, d))])
    if not runs:
        print("No runs found in", logdir)
        return

    # If user passed --runs, parse it (either indices like '2,4' or names 'runA,runB')
    if runs_arg:
        parts = [p.strip() for p in runs_arg.split(',') if p.strip()]
        if len(parts) != 2:
            print("--runs expects exactly two comma-separated values (indices or run names)")
            return
        chosen = []
        for p in parts:
            # numeric -> index
            if p.isdigit():
                idx = int(p)
                if idx < 0 or idx >= len(runs):
                    print(f"Index out of range: {idx}")
                    return
                chosen.append(runs[idx])
            else:
                # match by name
                if p in runs:
                    chosen.append(p)
                else:
                    print(f"Run name not found: {p}")
                    return
    else:
        # interactive selection if more than 2 runs, otherwise use all
        if len(runs) > 2:
            print("Available runs:")
            for i, r in enumerate(runs):
                print(f"  [{i}] {r}")
            sel = input("Enter two indices separated by comma to plot (e.g. 0,3), or press Enter to plot the last two: ").strip()
            if sel:
                try:
                    parts = [int(x.strip()) for x in sel.replace(' ', ',').split(',') if x.strip()]
                    if len(parts) != 2:
                        print("Please enter exactly two indices.")
                        return
                    chosen = [runs[p] for p in parts]
                except Exception as e:
                    print("Invalid selection:", e)
                    return
            else:
                chosen = runs[-2:]
        else:
            chosen = runs

    plt.figure(figsize=(10,6))
    plotted = 0
    for run in chosen:
        path = os.path.join(logdir, run)
        try:
            series = load_series(path)
        except Exception as e:
            print(f"skip {run}: {e}")
            continue
        for tag, (steps, vals) in series.items():
            if not steps:
                continue
            # apply smoothing if requested
            plot_vals = vals
            try:
                plot_vals = smooth_values(vals, float(smooth))
            except Exception:
                pass
            label = f"{run} / {tag}"
            plt.plot(steps, plot_vals, label=label)
            plotted += 1
    if plotted == 0:
        print("No avg_episodes/episode_length series found in logs.")
        return
    plt.xlabel("Step")
    plt.ylabel("Episodes (avg or per-episode)")
    plt.title("Avg episodes / Episode lengths over time")
    plt.legend()
    plt.grid(True)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(out)
    print("Saved plot to", out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", default="logs", help="TensorBoard logs directory")
    parser.add_argument("--out", default="eval/results/avg_episodes.png", help="Output image path")
    parser.add_argument("--runs", help="Two runs to plot, comma-separated: either indices '2,4' or run names 'runA,runB'")
    parser.add_argument("--smooth", type=float, default=0.0, help="Smoothing factor in [0,1], like TensorBoard (0=no smoothing)")
    args = parser.parse_args()
    main(args.logdir, args.out, runs_arg=args.runs, smooth=args.smooth)
