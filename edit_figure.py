import pickle
import os
from matplotlib import pyplot as plt
import argparse


def main(name):
    # load he pickle
    _path = os.path.join("experiments", "figures")
    objname = f"{name}.pkl"
    figname_png = f"{name}_modified.png"
    figname_eps = f"{name}_modified.pdf"
    file = open(os.path.join(_path, objname), 'rb')
    plot = pickle.load(file)

    # changing title if you want
    # plot.set_title(f"MODIFIED")

    # rotate the xlabel so that it does not overlap

    # new_x_tick = [i for i in range(x_tick_min, x_tick_max, 500)]
    plot.set_xticklabels(plot.get_xticks(), rotation=45)
    for ind, label in enumerate(plot.get_xticklabels()):
        if ind % 10 == 0:  # every 10th label is kept
            label.set_visible(True)
        else:
            label.set_visible(False)
    plt.ylim(-1000, 400)

    # add padding on bottom and left so that xlabel and ylabel do not get cut off
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.gcf().subplots_adjust(left=0.15)

    # add title for the legend. For some reason, it was not there originally
    plot.legend().set_title("Expert Ratio")
    path_png = os.path.join(_path, figname_png)
    plot.figure.savefig(path_png)

    path_eps = os.path.join(_path, figname_eps)
    # plot.set_rasterized(True)
    plot.figure.savefig(path_eps)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", type=str, default="CQLSAC", help="the name pkl file you want to load and edit")
    args = vars(ap.parse_args())
    main(name=args["name"])
