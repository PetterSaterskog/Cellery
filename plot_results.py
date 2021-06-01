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
patientList = ['patient1','patient13','patient14', 'patient36', 'patient39']
patientList = ['patient1','patient2','patient13','patient14','patient23','patient36','patient37', 'patient39']

for fn in os.listdir(inputDir):
	name = fn[:-19]
	if name in patientList:
		im = keren.loadImage(fn)
		origW, origCells = im.W, im.cells
		with open(f'{name}.pickle', 'rb') as h:
			genW, genCells = pickle.load(h)

		with open(f'{name}_meanS0=True.pickle', 'rb') as h:
			genCommonS0W, genCommonS0Cells = pickle.load(h)

		# pl.figure(figsize=(15, 15))
		fig, axs = pl.subplots(3, figsize=(9, 3*9))
		fig.tight_layout()
		# fig.suptitle('Vertically stacked subplots')

		for ax in axs: ax.axis('scaled')

		pl.sca(axs[0])
		pl.title(f'Original image, {sum(len(origCells[t]) for t in origCells)} cells')
		plotCells(origCells, origW)
		pl.legend(loc = 'upper right', framealpha=0.7)

		pl.sca(axs[1])
		pl.title(f'Randomly generated cell distribution with cell count\n matching {name} and pair-correlator matching patient-average pair correlator.')
		plotCells(genCommonS0Cells, genCommonS0W)

		pl.sca(axs[2])
		pl.title(f'Randomly generated cell distribution with cell count\n and pair-correlator matching image from {name}.')
		plotCells(genCells, genW)
		# pl.show()
		pl.savefig(f"figs/{name}_triplet.pdf")
		pl.close()
