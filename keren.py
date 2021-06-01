from cell_distribution import CellDistribution
from collections import defaultdict
import numpy as np

inputDir = "keren_cell_positions_data"

def loadImage(fileName, L=800, margin = 15):
	cells = defaultdict(list)
	with open(inputDir+'/'+fileName) as f:
		f.readline() # skip header
		while True:
			line = f.readline()
			if line:
				cols = line.split(',')
				if cols[3] in {'Kreatin-positive tumor', 'CD4-T'}:
					cells[cols[3]].append((float(cols[0]), float(cols[1])))
			else: break
	for t in cells:
		ps = np.array(cells[t])
		inside = (ps[:,0]>margin) & (ps[:,0] < L-margin) & (ps[:,1]>margin) & (ps[:,1] < L-margin)
		# cells[t] = ps[ np.where(inside) ] - margin
		cells[t] = ps[ inside ] - margin
	return CellDistribution(cells, L-2*margin)
