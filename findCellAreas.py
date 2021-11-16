import numpy as np
from spencerNewUtils import *
import matplotlib.pyplot as pl

cellsByType = loadTumor(f"images/spencer/new.csv")


cellsByTypeJoined = {k:cellsByType[k] for k in cellsByType if not k in tumorCells}
cellsByTypeJoined['Tumor'] = np.concatenate([cellsByType[k] for k in tumorCells])

rs = np.geomspace(10, 300, 10)
areas = []
for r in rs:
	areas.append(findMaxCellAreas(cellsByType, minRadius=r, maxRadius=r))
for t in areas[0]:
	pl.semilogx(rs, [2*np.sqrt(a[t]/np.pi) for a in areas], label=t, marker='.')
pl.ylabel("Cell diameter")
pl.legend()
pl.show()


# print({k:np.sqrt(a/np.pi) for k,a in areas.items()})