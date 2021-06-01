import numpy as np
import matplotlib.pyplot as pl
import os
from tqdm import tqdm
from collections import defaultdict
import matplotlib.patches as patches
from multiprocessing import Pool
from plot_cells import plotCells, plotHist
import keren
from cell_distribution import generateCellDistribution, CellDistribution
import pickle

inputDir = "keren_cell_positions_data"

images = {fn[:-19]:keren.loadImage(fn) for fn in os.listdir(inputDir)}

categories = {'Tumor':{'Kreatin-positive tumor','Tumor'},
				'DC / Mono / Neu / Unidentified / Other immune':{'Mono / Neu','DC / Mono','DC','Unidentified', 'Other immune'},
				'CD8-T':{'CD8-T'},
				'CD4-T':{'CD4-T'},
				'CD3-T':{'CD3-T'},
				'B':{'B'},
				'NK':{'NK'},
				'Neutrophils':{'Neutrophils'},
				'Tregs':{'Tregs'},
				'Macrophages':{'Macrophages'},
				'Endothelial':{'Endothelial'},
				'Mesenchymal-like':{'Mesenchymal-like'}
				}

#TLS: 35, 28, 32, 16

counts = {n:{cat:sum(len(images[n].cells[t]) if t in images[n].cells else 0 for t in categories[cat]) for cat in categories} for n in images}

countVecs = np.array([[counts[n][c] if c in counts[n] else 0 for c in categories.keys()] for n in counts])
countVecsNorm = countVecs # / np.sum(countVecs, axis = 1)[:, np.newaxis]

countVecsStd = (countVecsNorm-np.mean(countVecsNorm, axis=0)) / np.std(countVecsNorm, axis=0)

cov = np.cov(countVecsStd.T)
l, v = np.linalg.eigh(cov)
plotPoints = countVecsStd.dot(v[::-1])
print(l)
print(v)

pl.figure()
pl.plot(plotPoints[:,0], plotPoints[:,1], linestyle = 'None', marker = '.')
# pl.plot(countVecsStd.dot( v[:, -2]), countVecsStd.dot( v[:, -3]), linestyle = 'None', marker = '.')

for i, txt in enumerate(images):
    pl.gca().annotate(txt, plotPoints[i,:2])

fig = pl.figure()
ax = fig.add_subplot(projection='3d')


ax.scatter(plotPoints[:,0],plotPoints[:,1],plotPoints[:,2], marker='.')

ax.set_xlabel('0')
ax.set_ylabel('1')
ax.set_zlabel('2')

controlVecsStd = np.random.normal(size=countVecsStd.shape)
cov = np.cov(controlVecsStd.T)
l, v = np.linalg.eigh(cov)
print(l)
print(v)

print(list(zip(categories.keys(), v[:, -1])))

print(list(zip(categories.keys(), v[:, -2])))
pl.show()