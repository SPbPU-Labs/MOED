import os

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from src.model import Model


N = [100, 1000, 10000]
R = 100


def test_noise_methods(N: int):
    for data, i in zip([Model.noise(N, R), Model.myNoise(N, R)], range(1, 3)):
        plt.subplot(1, 2, i)
        line_plot = sns.lineplot(x=np.arange(N), y=data)
        if i % 2 != 0:
            plt.title('Noise method')
        else:
            plt.title('My_Noise method')
        line_plot.set_xlabel('t', fontsize=12)
        line_plot.set_ylabel('R', fontsize=12)


if __name__ == '__main__':
    sns.set(style="whitegrid")
    if not os.path.exists("./plots"):
        os.makedirs("./plots")
    for i in range(len(N)):
        fig = plt.figure(figsize=(14, 8))
        fig.suptitle(f"Page #{i+1}")
        test_noise_methods(N[i])
        fig.savefig(f"./plots/fig_{i+1}.png")

    plt.tight_layout()
    plt.show()
