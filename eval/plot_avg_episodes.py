#!/usr/bin/env python3
"""
Scan TensorBoard logs under `logs/` and plot episode-length / avg_episodes series.
By default only runs whose names contain an agent name from `src/agents/` are considered,
and for each agent the newest run directory (most recently modified) is plotted.
Saves to a PNG file.
"""
import os
import argparse
import matplotlib.pyplot as plt

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except Exception:
    EventAccumulator = None


def find_agent_names():
    """Return list of agent folder names by inspecting likely src/agents paths."""
    candidates = [
        os.path.join(os.getcwd(), 'src', 'agents'),
        os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src', 'agents'),
        os.path.join(os.path.dirname(__file__), '..', 'src', 'agents')
    ]
    seen = set()
    for p in candidates:
        p = os.path.abspath(p)
        if os.path.isdir(p):
            try:
                for name in os.listdir(p):
                    full = os.path.join(p, name)
                    if os.path.isdir(full):
                        seen.add(name.lower())
            except Exception:
                pass
    return sorted(seen)


def load_series(run_dir):
    if EventAccumulator is None:
        raise RuntimeError("tensorboard is required to read event files (pip install tensorboard)")
    ea = EventAccumulator(run_dir, size_guidance={"scalars": 0})
    ea.Reload()
    tags = ea.Tags().get('scalars', [])
    series = {}
    for tag in tags:
        # collect only avg_episodes series
        if 'avg_episodes' in tag:
            events = ea.Scalars(tag)
            steps = [e.step for e in events]
            vals = [e.value for e in events]
            series[tag] = (steps, vals)
    return series


def smooth_values(vals, smooth_factor: float):
    """Apply exponential smoothing like TensorBoard: smooth in [0,1).
    Higher smooth_factor => smoother (0.9 means heavy smoothing).
    """
    try:
        alpha = float(smooth_factor)
    except Exception:
        return vals
    if not vals:
        return vals
    if alpha <= 0.0:
        return vals
    if alpha >= 1.0:
        alpha = 0.999
    out = []
    last = float(vals[0])
    for v in vals:
        v = float(v)
        last = last * alpha + v * (1.0 - alpha)
        out.append(last)
    return out


def choose_newest_per_agent(runs, logdir, agent_names):
    """Return a list of runs: for each agent name pick the newest run that contains it."""
    chosen = []
    # determine newest based on the newest events.out* file inside each run dir
    run_mtimes = []
    for r in runs:
        full = os.path.join(logdir, r)
        newest = 0
        try:
            for fname in os.listdir(full):
                if fname.startswith('events.out'):
                    fpath = os.path.join(full, fname)
                    try:
                        m = os.path.getmtime(fpath)
                        if m > newest:
                            newest = m
                    except Exception:
                        pass
        except Exception:
            pass
        # fallback to dir mtime when no events files found
        if newest == 0:
            try:
                newest = os.path.getmtime(full)
            except Exception:
                newest = 0
        run_mtimes.append((r, newest))

    for agent in agent_names:
        matches = [(r, mt) for (r, mt) in run_mtimes if agent in r.lower()]
        if matches:
            matches.sort(key=lambda x: x[1], reverse=True)
            chosen.append(matches[0][0])
    # deduplicate while preserving order
    seen = set()
    out = []
    for r in chosen:
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out


def main(logdir, out, runs_arg=None, smooth=0.0):
    if not os.path.isdir(logdir):
        print("Log directory not found:", logdir)
        return
    runs = sorted([d for d in os.listdir(logdir) if os.path.isdir(os.path.join(logdir, d))])
    if not runs:
        print("No runs found in", logdir)
        return

    agent_names = find_agent_names()

    # If user passed --runs, parse it (indices or names) and honor that selection
    if runs_arg:
        parts = [p.strip() for p in runs_arg.split(',') if p.strip()]
        chosen = []
        for p in parts:
            if p.lstrip('-').isdigit():
                idx = int(p)
                if 0 <= idx < len(runs):
                    chosen.append(runs[idx])
                elif 1 <= idx <= len(runs):
                    chosen.append(runs[idx-1])
                else:
                    print(f"Index out of range: {idx}")
                    return
            else:
                if p in runs:
                    chosen.append(p)
                else:
                    # allow partial match
                    found = [r for r in runs if p.lower() in r.lower()]
                    if found:
                        chosen.extend(found)
                    else:
                        print(f"Run name not found: {p}")
                        return
    else:
        # default: choose newest run per agent if agents detected
        if agent_names:
            chosen = choose_newest_per_agent(runs, logdir, agent_names)
            if not chosen:
                # fallback: last two runs
                chosen = runs[-2:]
        else:
            # no agents folder found: interactive/fallback behavior
            if len(runs) > 2:
                print("Available runs:")
                for i, r in enumerate(runs):
                    print(f"  [{i}] {r}")
                sel = input("Enter two indices separated by comma to plot (e.g. 0,3), or press Enter to plot the last two: ").strip()
                if sel:
                    try:
                        parts = [int(x.strip()) for x in sel.replace(' ', ',').split(',') if x.strip()]
                        chosen = [runs[p] for p in parts]
                    except Exception as e:
                        print("Invalid selection:", e)
                        return
                else:
                    chosen = runs[-2:]
            else:
                chosen = runs

    plt.figure(figsize=(10, 6))
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
            plot_vals = vals
            try:
                plot_vals = smooth_values(vals, float(smooth))
            except Exception:
                pass
            short = 'avg' if 'avg_episodes' in tag else 'episode'
            label = f"{run} ({short})"
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
    parser.add_argument("--runs", help="Comma-separated runs or indices to plot (overrides agent-auto-selection)")
    parser.add_argument("--smooth", type=float, default=0.0, help="Smoothing factor in [0,1), like TensorBoard (0=no smoothing)")
    args = parser.parse_args()
    main(args.logdir, args.out, runs_arg=args.runs, smooth=args.smooth)
