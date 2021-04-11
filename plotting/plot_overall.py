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


f, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, sharey=True)

# df = pd.read_csv("cqlsac_0.csv")
df = smooth("cqlsac_0.csv")
df2 = smooth("cqlsac_1.csv")

df3 = smooth("bc_0.csv")
df4 = smooth("bc_1.csv")

df.insert(0, "Agent", ["CQLSAC" for _ in range(len(df.index))], allow_duplicates=False)
df.insert(0, "Ratio", ["0.0" for _ in range(len(df.index))], allow_duplicates=False)

df2.insert(0, "Agent", ["CQLSAC" for _ in range(len(df.index))], allow_duplicates=False)
df2.insert(0, "Ratio", ["1.0" for _ in range(len(df.index))], allow_duplicates=False)

df3.insert(0, "Agent", ["BC" for _ in range(len(df.index))], allow_duplicates=False)
df3.insert(0, "Ratio", ["0.0" for _ in range(len(df.index))], allow_duplicates=False)

df4.insert(0, "Agent", ["BC" for _ in range(len(df.index))], allow_duplicates=False)
df4.insert(0, "Ratio", ["1.0" for _ in range(len(df.index))], allow_duplicates=False)
# df_main = pd.DataFrame()

df_main = pd.concat([df, df2, df3, df4], axis=0)
# print(df.head())
# print(df2.head())

# df_main.insert(0, "Value2", df["Value"], allow_duplicates=False)
# df_main.insert(0, "Value1", df["Value"], allow_duplicates=False)
# df_main.insert(0, "Step",df["Step"], allow_duplicates=False)

print(len(df.index))
print(len(df2.index))

print(df_main.head())
print(len(df_main.index))

sns.lineplot(data=df_main, x="Step", y="Value", hue="Ratio", style="Agent", palette=sns.color_palette("hls", 2),
             markers=True, ax=ax1)

sns.lineplot(data=df_main, x="Step", y="Value", hue="Ratio", style="Agent", palette=sns.color_palette("hls", 2),
             markers=True, ax=ax2, markevery=20)

# x=[ep for ep in range(0, max(episodes) + log_freq, log_freq)]
# y=[0 for _ in range(0, max(episodes) + log_freq, log_freq)]

# sns.lineplot(x=[5000, 5000], y=[-400, 300], color="black", linestyle="--")
# plt.plot(x = [5000,5000], y = [-400, 300], color="black")
print(max(df["Step"]))

ax1.set_xlim(0, 4800)
ax2.set_xlim(4800, max(df["Step"]) + 75)
ax2.axvline(x=4990, color=(0, 0, 0, 0.5), linestyle='-.')
ax2.get_yaxis().set_visible(False)

f.subplots_adjust(wspace=0.0)
plt.ylim(-400, 300)
f.show()

# p = sns.color_palette("Blues", 1)
# colors = []
# for _p in p:
#     print(_p)
#     c = (0, _p[1], _p[2], 1)
#     colors.append(c)
#
# print(colors)
#
# sns.set_palette(colors)
# pal = sns.color_palette(colors)
#
#
# # pal = sns.color_palette(pal)
# sns.lineplot(data=df, x="Step", y="Value",color=(255,0,0))
# sns.lineplot(data=df2, x="Step", y="Value", color=(0,0,0))
# plt.show()
#
# # print(p)
