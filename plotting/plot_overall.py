import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def smooth(csv_path, weight=0.96):
    data = pd.read_csv(filepath_or_buffer=csv_path, header=0, names=['Step', 'Value'],
                       dtype={'Step': np.int, 'Value': np.float})
    scalar = data['Value'].values
    last = scalar[0]
    smoothed = []
    for point in scalar:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val

    save = pd.DataFrame({'Step': data['Step'].values, 'Value': smoothed})
    return save
    # save.to_csv('smooth_' + csv_path)


f, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, gridspec_kw={ 'top':0.92,  'wspace': 0, 'hspace': 0, 'right': 0.96}, sharey=True)

df_cql_0 = smooth("cql_0.csv")
df_cql_1 = smooth("cql_1.csv")

df_bc_0 = smooth("bc_0.csv")
df_bc_1 = smooth("bc_1.csv")

df_sacoff_0 = smooth("sacoff_0.csv")
df_sacoff_1 = smooth("sacoff_1.csv")

df_sac_bl = smooth("sac_baseline.csv")

df_cql_0.insert(0, "Agent", ["CQLSAC" for _ in range(len(df_cql_0.index))], allow_duplicates=False)
df_cql_0.insert(0, "Ratio", ["0.0" for _ in range(len(df_cql_0.index))], allow_duplicates=False)
df_cql_1.insert(0, "Agent", ["CQLSAC" for _ in range(len(df_cql_1.index))], allow_duplicates=False)
df_cql_1.insert(0, "Ratio", ["1.0" for _ in range(len(df_cql_1.index))], allow_duplicates=False)

df_bc_0.insert(0, "Agent", ["BC" for _ in range(len(df_bc_0.index))], allow_duplicates=False)
df_bc_0.insert(0, "Ratio", ["0.0" for _ in range(len(df_bc_0.index))], allow_duplicates=False)
df_bc_1.insert(0, "Agent", ["BC" for _ in range(len(df_bc_1.index))], allow_duplicates=False)
df_bc_1.insert(0, "Ratio", ["1.0" for _ in range(len(df_bc_1.index))], allow_duplicates=False)

df_sacoff_0.insert(0, "Agent", ["SACOffline" for _ in range(len(df_sacoff_0.index))], allow_duplicates=False)
df_sacoff_0.insert(0, "Ratio", ["0.0" for _ in range(len(df_sacoff_0.index))], allow_duplicates=False)
df_sacoff_1.insert(0, "Agent", ["SACOffline" for _ in range(len(df_sacoff_1.index))], allow_duplicates=False)
df_sacoff_1.insert(0, "Ratio", ["1.0" for _ in range(len(df_sacoff_1.index))], allow_duplicates=False)

df_sac_bl.insert(0, "Agent", ["SAC Baseline" for _ in range(len(df_sac_bl.index))], allow_duplicates=False)
df_sac_bl.insert(0, "Ratio", ["1.0" for _ in range(len(df_sacoff_1.index))], allow_duplicates=False)

df_main = pd.concat([df_cql_0, df_cql_1, df_bc_0, df_bc_1, df_sacoff_0, df_sacoff_1, df_sac_bl], axis=0)

df_sac_bl.plot.line(x='Step', y='Value', ax=ax1, color="dimgray", alpha=1)
df_bc_0.plot.line(x='Step', y='Value', ax=ax1, color="dodgerblue", alpha=1)
df_bc_1.plot.line(x='Step', y='Value', ax=ax1, color="dodgerblue", alpha=0.5)
df_cql_0.plot.line(x='Step', y='Value', ax=ax1, color="red", alpha=1)
df_cql_1.plot.line(x='Step', y='Value', ax=ax1, color="red", alpha=0.5)
df_sacoff_0.plot.line(x='Step', y='Value', ax=ax1, color="forestgreen", alpha=1)
df_sacoff_1.plot.line(x='Step', y='Value', ax=ax1, color="forestgreen", alpha=0.5)

df_sac_bl.plot.line(x='Step', y='Value', ax=ax2, color="dimgray", alpha=1)
df_bc_0.plot.line(x='Step', y='Value', ax=ax2, color="dodgerblue", alpha=1)
df_bc_1.plot.line(x='Step', y='Value', ax=ax2, color="dodgerblue", alpha=0.5)
df_cql_0.plot.line(x='Step', y='Value', ax=ax2, color="red", alpha=1)
df_cql_1.plot.line(x='Step', y='Value', ax=ax2, color="red", alpha=0.5)
df_sacoff_0.plot.line(x='Step', y='Value', ax=ax2, color="forestgreen", alpha=1)
df_sacoff_1.plot.line(x='Step', y='Value', ax=ax2, color="forestgreen", alpha=0.5)

labels = ["SAC (online)", "BC 0.0", "BC 1.0", "CQLSAC 0.0", "CQLSAC 1.0", "SAC-Off 0.0", "SAC-Off 1.0"]
ax2.legend(labels=labels)

XMIN = 4700
XMAX = 6000
ax1.set_xlim(0, XMIN)
ax2.set_xlim(XMIN, XMAX)

ax1.annotate("Offline",
             xy=(1100, 350),
             xytext=(1100, 350),
             arrowprops={"edgecolor": "dimgray", "ls": '-', "lw": 0, "arrowstyle": '-|>'},
             bbox={"boxstyle": "round", "pad": 0.4, "facecolor": "oldlace", "edgecolor": "tan"},
             ha="center", fontsize=15,
             color="black", alpha=1
             )

ax2.annotate("Online",
             xy=(5350, 350),
             xytext=(5350, 350),
             arrowprops={"edgecolor": "dimgray", "ls": '-', "lw": 0, "arrowstyle": '-|>'},
             bbox={"boxstyle": "round", "pad": 0.4, "facecolor": "thistle", "edgecolor": "palevioletred"},
             ha="center", fontsize=15,
             color="black", alpha=1
             )

vertical_line = 4990

ax1.axvspan(0, XMIN, fc="lightyellow", alpha=0.3)
ax2.axvspan(XMIN, vertical_line, fc="lightyellow", alpha=0.3)
ax2.axvspan(vertical_line, XMAX, fc="plum", alpha=0.2)

ax1.legend().set_visible(False)
ax1.spines['right'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax2.get_yaxis().set_visible(False)
ax1.xaxis.label.set_visible(False)
ax2.xaxis.label.set_visible(False)

ax2.axvline(x=vertical_line, color=(0, 0, 0, 0.5), linestyle='-.')

YMIN = -700
YMAX = 450
plt.ylim(YMIN, YMAX)
plt.suptitle('Scores of each agent during training', fontsize=16)
f.text(0.52, 0.03, 'Episode', ha='center', fontsize=10)
f.text(0.03, 0.5, 'Episodic Score', va='center', rotation='vertical', fontsize=10)
# add padding on bottom and left so that xlabel and ylabel do not get cut off
# f.show()
f.savefig("training_overall.pdf")

