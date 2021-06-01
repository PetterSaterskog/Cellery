


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


# compartmentalized, would be interesting to see what percolation phase these are in
patients = [3,4,5,6,7,10,13,16,17,41]

cancerCellTypes = {'Kreatin-positive tumor','Tumor'}

for fn in os.listdir(inputDir):
	name = fn[:-19]
	if int(name[7:]) in patients:
		im = keren.loadImage(fn)
		pl.figure(figsize=(9, 9))
			
		pl.title(f'{name}, {sum(len(im.cells[t]) for t in im.cells)} cells')

		for t in im.cells:
			pl.plot(im.cells[t][:,0], im.cells[t][:,1], '.', color=(1,0,0) if t in cancerCellTypes else (0,0.8,0))
		# pl.legend(loc = 'upper right', framealpha=0.7)

pl.show()
	# pl.savefig(f"figs/cell_distributions/{name}.pdf")
	# pl.close()

