import numpy as np
import matplotlib.pyplot as pl
import os
from tqdm import tqdm
from collections import defaultdict
import matplotlib.patches as patches

inputDir = "cell_positions_data"
figDir = "figs/"

cells = []
types = {}

def typeId(t):
	if not t in types:
		types[t] = len(types)
	return types[t] 

L = 800
margin = 18
names = []
for f in os.listdir(inputDir):
	names.append(f[:-19])
	print("{}: ".format(len(cells))+f)
	rows = np.loadtxt(inputDir+'/'+f, delimiter=',', skiprows=1, converters = {3: lambda s: typeId(s.decode("utf-8"))})
	inside = (rows[:,0]>margin)*(rows[:,0] < L-margin)*(rows[:,1]>margin)*(rows[:,1] < L-margin)
	print('{:.1f}% inside margins.'.format(100*np.sum(inside)/rows.shape[0]))
	cells.append(rows[ np.where(inside) ]) #filter out everything outside margin

cellsByType = []

for p in cells:
	d = [[] for i in types]
	for c in p:
		d[int(c[3])].append(c[:2])
	for i in range(len(types)):
		d[i] = np.array(d[i]) if len(d[i])>0 else np.zeros((0,2))
	cellsByType.append(d)

# print(types)

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

TCellTypes = {'CD8-T','CD4-T','CD3-T','Tregs'}
BCellTypes = {'B'}
otherImmuneCellTypes = {'Macrophages', 'Other immune', 'Mono / Neu','DC / Mono','Neutrophils','DC','NK'}
cancerCellTypes = {'Kreatin-positive tumor','Tumor'}
otherCellTypes = {'Endothelial','Mesenchymal-like'}
unknownCellTypes = {'Unidentified'}

for i in range(len(cellsByType)):

	pl.figure(figsize=(15, 15))
	for typeName in types:
		pCells = cellsByType[i][typeId(typeName)]
		if typeName in TCellTypes:
			marker = 'o'
		elif typeName in BCellTypes:
			marker = '1'
		elif typeName in otherImmuneCellTypes:
			marker = '^'
		elif typeName in cancerCellTypes:
			marker = '*'
		elif typeName in otherCellTypes:
			marker = 'x'
		elif typeName in unknownCellTypes:
			marker = '$?$'
		else:
			print(typeName)
			print(cancerCells)
			assert(False)
		pl.plot(pCells[:,0], pCells[:,1],linestyle="None", marker=marker, label = typeName)

	pl.gca().add_patch(patches.Rectangle((margin, margin), L-2*margin, L-2*margin,  linewidth=1, edgecolor='r', facecolor='none'))
	pl.legend()
	pl.savefig(figDir+f"cellsByType_{names[i]}.pdf")
	pl.close()
