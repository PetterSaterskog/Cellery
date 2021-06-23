import numpy as np
import matplotlib.pyplot as pl
import os
from tqdm import tqdm
from collections import defaultdict
import matplotlib.patches as patches
from multiprocessing import Pool
from plot_cells import plotCells, plotHist, typeOrder
import keren
from cell_distribution import generateCellDistribution, CellDistribution, radialFourierTransformRToQ
import pickle
from matplotlib.colors import hsv_to_rgb

inputDir = "keren_cell_positions_data"

for fn in os.listdir(inputDir):
	name = fn[:-19]
	if not name == 'patient32':#'patient17':
		continue
	
	if False:
		W = 1200
		n = 800
		ps = []
		for i in range(n):
			while True:
				p = np.random.rand(2)*W
				tooClose = False
				for j in range(len(ps)):
					if np.linalg.norm(ps[j]-p)<10:
						tooClose = True
						break
				if not tooClose:
					ps.append(p)
					break
		im = CellDistribution({'1':np.array(ps), '2':np.random.rand(n, 2)*W}, W, torus=False)
	else:
		im = keren.loadImage(fn, threshold = 400)

	im.z = 20 #15 #um			
	
	dr = 1.5
	n = 200
	dq=np.pi/n/dr
	rs = (0.5+np.arange(n)) * dr

	g = im.getG(dr, n)
	c = im.getC(dr, n)
	cReg = im.getC(dr, n, longRangeReg = 1e-15*0.00001, highFreqReg = 0*3e-2)
	
	pl.figure( figsize=(10, 10))
	pl.title("c (-- without regulator)")
	pl.grid()
	for i, p in enumerate(c):
		col = hsv_to_rgb((i/len(c),1,1))
		pl.plot(rs, cReg[p], linestyle='-', color=col, label=p)
		pl.plot(rs, c[p], linestyle='--', color=col)

	pl.legend()

	gFromcReg = im.getGFromC(cReg, dr, n)

	pl.figure( figsize=(10, 10))
	pl.title("g from c (-- original)")

	for i,p in enumerate(g):
		col = hsv_to_rgb((i/len(g),1,1))
		pl.plot(rs, gFromcReg[p][1], linestyle='-', label=p, color=col)
		pl.plot(rs, g[p], linestyle='--', color=col)

	pl.legend()


	pl.show()
	exit(0)
	
	pl.savefig(f"figs/correlators/radia_distrib/{name}.pdf")
	pl.close()
		

	edges = [0, 20, 75, 150]

	corr, _ = im.getCorrs(edges)
	
	
	for ri in range(len(edges)-1):
		def pair(t1,t2):
			return (min(t1,t2), max(t1,t2))

		corrMat = np.array([[corr[pair(t1,t2)][ri] if pair(t1,t2) in corr else 0 for t2 in filteredTypeOrder] for t1 in filteredTypeOrder])

		def plotCorrMat(cm, name):
			pl.figure( figsize=(12, 12))
			pl.imshow(cm, vmin=0, vmax=5)
			ax = pl.gca()

			ax.set_xticks(np.arange(len(filteredTypeOrder)))
			ax.set_yticks(np.arange(len(filteredTypeOrder)))
			# ... and label them with the respective list entries
			ax.set_xticklabels(filteredTypeOrder)
			ax.set_yticklabels(filteredTypeOrder)

			# Rotate the tick labels and set their alignment.
			pl.setp(ax.get_xticklabels(), rotation=45, ha="right",
					rotation_mode="anchor")

			# Loop over data dimensions and create text annotations.
			for i in range(len(filteredTypeOrder)):
				for j in range(len(filteredTypeOrder)):
					text = ax.text(j, i, f"{cm[i,j]:.1f}",
								ha="center", va="center", color="w")

			pl.savefig(f"figs/correlators/matrices/{name}.pdf")
			pl.close()
		plotCorrMat(corrMat, f"{name}_{ri}")

		eigVal, eigVec = np.linalg.eigh(corrMat-1)
		eigOrder = np.argsort(-np.abs(eigVal))
		eigVal = eigVal[eigOrder]
		eigVec = eigVec[:, eigOrder]
		print(f"{name} eigenvalues: {eigVal}")
		appCorrMat = 1
		nEig = 2
		for i in range(nEig):
			appCorrMat += eigVal[i]*np.outer(eigVec[:,i], eigVec[:,i])			

		
		plotCorrMat(appCorrMat, f"{name}_{ri}_{nEig}eig")
		if ri==2:
			U = np.sqrt(-eigVal[0])*eigVec[:,0]
			V = np.sqrt(eigVal[1])*eigVec[:,1]
			A = U + V
			E = U - V
			pl.figure()
			pl.gca().bar(filteredTypeOrder,A)
			pl.setp(pl.gca().get_xticklabels(), rotation=45, ha="right",
						rotation_mode="anchor")
			pl.figure()
			pl.gca().bar(filteredTypeOrder,E)
			pl.setp(pl.gca().get_xticklabels(), rotation=45, ha="right",
						rotation_mode="anchor")

			pl.show()
	exit(0)