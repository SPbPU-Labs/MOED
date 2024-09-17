import os

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from src.model import Model


N = 1000
M = 5
R = 10
Rs = R * 0.1


def test_spikes_method():
    data = Model.spikes(N, M, R, Rs)
    line_plot = sns.lineplot(x=np.arange(N), y=data)
    plt.title('Spikes method')
    line_plot.set_xlabel('t', fontsize=12)
    line_plot.set_ylabel('R', fontsize=12)


if __name__ == '__main__':
    sns.set(style="whitegrid")
    if not os.path.exists("./plots"):
        os.makedirs("./plots")

    fig = plt.figure(figsize=(14, 8))
    test_spikes_method()
    fig.savefig("./plots/fig.png")
    plt.tight_layout()
    plt.show()
