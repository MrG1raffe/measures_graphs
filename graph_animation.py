import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import List


def graph_animation(graphs: List, interval: int = 300, ax = None, fig = None):
    if (ax is None) or (fig is None):
        fig, ax = plt.subplots(figsize=(10, 6))
    def update(frame):
        ax.clear()
        graphs[frame].plot(ax=ax)
        ax.set_title(f"Graph Evolution: Step {frame}")

    # Create the animation
    return FuncAnimation(fig, update, frames=len(graphs), interval=interval)