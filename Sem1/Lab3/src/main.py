import os

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from src.model import Model


N = 1000
M = [i for i in range(1, 11)]
R = 10
Rs = R * 0.1


def test_spikes_method():
    for i in range(len(M)):
        data = Model.spikes(N, M[i], R, Rs)
        plt.subplot(4, 3, i+1)
        line_plot = sns.lineplot(x=np.arange(N), y=data)
        plt.title(f"Spikes method. M = {M[i]}")
        line_plot.set_xlabel('t', fontsize=12)
        line_plot.set_ylabel('R', fontsize=12)


if __name__ == '__main__':
    sns.set(style="whitegrid")
    if not os.path.exists("./plots"):
        os.makedirs("./plots")

    fig = plt.figure(figsize=(12, 16))
    test_spikes_method()
    plt.tight_layout(pad=1.7)
    plt.show()
    fig.savefig("./plots/fig.png")
