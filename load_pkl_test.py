import pickle
import os

_path = os.path.join("experiments", "figures")

objname = "CQLSAC.pkl"
figname = "CQLSAC_modified.png"

file = open(os.path.join(_path, objname), 'rb')
plot = pickle.load(file)
plot.set_title(f"MODIFIED")
path = os.path.join(_path, figname)
plot.figure.savefig(path)
