import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def make_line_plot(title: str, x: np.ndarray, y: np.ndarray, x_label="Time [sec]", y_label="Amplitude"):
    plot = sns.lineplot(x=x, y=y)
    plot.set_xlabel(x_label, fontsize=12)
    plot.set_ylabel(y_label, fontsize=12)
    plt.title(title)
    return plt
    
def read_data(filename):
    with open(filename, 'rb') as f:
        data = np.fromfile(f, dtype=np.float32)
    return data

def rw(c, n, N):
    window = np.ones(N)
    for i in range(1, len(n), 2):
        window[n[i-1]:n[i]] = c[(i-1)//2]
    return window
