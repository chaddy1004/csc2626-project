import pickle
import os
from matplotlib import pyplot as plt

# load he pickle
_path = os.path.join("experiments", "figures")
objname = "CQLSAC.pkl"
figname = "CQLSAC_modified_bigger.png"
file = open(os.path.join(_path, objname), 'rb')
plot = pickle.load(file)

# changing title if you want
plot.set_title(f"MODIFIED")

# rotate the xlabel so that it does not overlap
plot.set_xticklabels(plot.get_xticks(), rotation=45)

# add padding on bottom and left so that xlabel and ylabel do not get cut off
plt.gcf().subplots_adjust(bottom=0.15)
plt.gcf().subplots_adjust(left=0.15)

# add title for the legend. For some reason, it was not there originally
plot.legend().set_title("Expert Ratio")
path = os.path.join(_path, figname)
plot.figure.savefig(path)
