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

for fn in os.listdir(inputDir):
	name = fn[:-19]
	
	im = keren.loadImage(fn)
	pl.figure(figsize=(9, 9))
		
	pl.title(f'{name}, {sum(len(im.cells[t]) for t in im.cells)} cells')
	plotCells(im.cells, im.W)
	pl.legend(loc = 'upper right', framealpha=0.7)
	pl.savefig(f"figs/cell_distributions/{name}.pdf")
	pl.close()

