import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def smooth(csv_path, weight=0.84):
    data = pd.read_csv(filepath_or_buffer=csv_path, header=0, names=['Step', 'Value'],
                       dtype={'Step': np.int, 'Value': np.float})
    scalar = data['Value'].values
    last = scalar[0]
    smoothed = []
    # Exponentially Weighted Moving Average filter
    # IIR first-order low pass filter
    for point in scalar:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val

    save = pd.DataFrame({'Step': data['Step'].values, 'Value': smoothed})
    return save
    # save.to_csv('smooth_' + csv_path)


ax = plt.gca()

df_sacoff_0 = smooth("sacoff_0.csv")
df_sacoff_1 = smooth("sacoff_1.csv")

df_sac_bl = smooth("sac_baseline.csv")

df_sacoff_0.insert(0, "Agent", ["SACOffline" for _ in range(len(df_sacoff_0.index))], allow_duplicates=False)
df_sacoff_0.insert(0, "Ratio", ["0.0" for _ in range(len(df_sacoff_0.index))], allow_duplicates=False)
df_sacoff_1.insert(0, "Agent", ["SACOffline" for _ in range(len(df_sacoff_1.index))], allow_duplicates=False)
df_sacoff_1.insert(0, "Ratio", ["1.0" for _ in range(len(df_sacoff_1.index))], allow_duplicates=False)

df_sac_bl.insert(0, "Agent", ["SAC Baseline" for _ in range(len(df_sac_bl.index))], allow_duplicates=False)
df_sac_bl.insert(0, "Ratio", ["1.0" for _ in range(len(df_sacoff_1.index))], allow_duplicates=False)

df_sac_bl.plot.line(x='Step', y='Value', ax=ax, color="dimgray", alpha=1)
df_sacoff_0.plot.line(x='Step', y='Value', ax=ax, color="forestgreen", alpha=1)
df_sacoff_1.plot.line(x='Step', y='Value', ax=ax, color="forestgreen", alpha=0.5)

labels = ["SAC (online)", "SAC-Off 0.0", "SAC-Off 1.0"]
ax.legend(labels=labels)

XMIN = 4700
XMAX = 6000
plt.xlim(XMIN, XMAX)
plt.axvline(x=5000, color=(0, 0, 0, 0.3), linestyle='-.')

ax.annotate("Offline",
            xy=(4850, 320),
            xytext=(4850, 320),
            arrowprops={"edgecolor": "dimgray", "ls": '-', "lw": 0, "arrowstyle": '-|>'},
            bbox={"boxstyle": "round", "pad": 0.4, "facecolor": "oldlace", "edgecolor": "tan"},
            ha="center", fontsize=12,
            color="black", alpha=1
            )

ax.annotate("Online",
            xy=(5150, 320),
            xytext=(5150, 320),
            arrowprops={"edgecolor": "dimgray", "ls": '-', "lw": 0, "arrowstyle": '-|>'},
            bbox={"boxstyle": "round", "pad": 0.4, "facecolor": "thistle", "edgecolor": "palevioletred"},
            ha="center", fontsize=12,
            color="black", alpha=1
            )

plt.axvspan(XMIN, 5000, fc="lightyellow", alpha=0.3)
plt.axvspan(5000, XMAX, fc="plum", alpha=0.2)

YMIN = -500
YMAX = 400
plt.title('Scores of SAC-off during training, detailed', fontsize=16)
plt.ylim(YMIN, YMAX)
plt.ylabel("Episodic Score")
plt.xlabel("Episode")
plt.savefig("sacoff_detailed.pdf")
# plt.show()
