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
		# im = keren.loadImage(fn, margin = 100)
		im = keren.loadImage(fn)

	im.z = 12 #um			
		# ps = {f"type{k}":np.random.rand(n, 2)*W for k in range(1)}
		
		# pl.figure( figsize=(10, 10))
		# for p in im.cells:
		# 	pl.plot(im.cells[p][:,0],im.cells[p][:,1], marker='.', linestyle='None')
		# pl.show()

	# filteredTypeOrder = [t for t in typeOrder if t in im.cells and len(im.cells[t])>=100]
	# resolution = 6 #3

	# minQ = (2*np.pi)/(0.9*im.W/2)
	# maxQ = 2*np.pi/resolution
	# qs = np.linspace(minQ, maxQ, max(2, int(2*maxQ//minQ)))

	if True:
		# res = 2# 4.0
		# qs = im.getQs(res)
		# S0, = im.getStructureFactors(qs)
		# H = im.getH(qs, S=S0)
		# i=0
		# pl.figure( figsize=(10, 10))
		# pl.title("S")
		# for p in S0:
		# 	if p[0]==p[1]:
		# 		pl.plot(qs, S0[p], label=p)
		# pl.legend()
		# pl.figure( figsize=(10, 10))
		# pl.title("H")
		# for p in S0:
		# 	if p[0]==p[1]:
		# 		pl.plot(qs, H[p], label=p)
		# pl.legend()
		# pl.savefig(f"figs/correlators/structure_factor/{name}.pdf")

		# pl.show()
		# exit(0)
		# sizes = {'CD8-T':5, 'Kreatin-positive tumor':8,'CD4-T':7, 'Macrophages':3}
		# ghs = im.getHardSphereG(sizes, res)
		# pl.figure( figsize=(10, 10))
		# pl.title("g hard sphere")
		# for p in ghs:
		# 	pl.plot(ghs[p][0], ghs[p][1], label=p)
		# pl.legend()
		dr = 1
		n = 400
		dq=np.pi/n/dr
		qs = (0.5+np.arange(n)) * dq

		# pl.figure( figsize=(10, 10))
		rs, g = im.getG(dr, n)
		# pl.title("g")
		# for p in g:
		# 	pl.plot(rs, g[p], label=p)
		# pl.legend()

		# pl.figure( figsize=(10, 10))
		_, c = im.getC(dr, n)
		_, cReg = im.getC(dr, n, longRangeReg = 0.00001, highFreqReg = 3e-2)

		# pl.title("c")
		# for p in c:
		# 	pl.plot(rs, c[p], label=p)
		# pl.legend()
		
		# gFromc = im.getGFromC(c, dr, n)
		# pl.figure( figsize=(10, 10))
		# pl.title("g from c")
		# for p in gFromc:
		# 	pl.plot(rs, gFromc[p][1], label=p)
		# pl.legend()
		
		H=im.getH(qs)

		rhos = im.getRhos()
		# rhoVec = np.array([rhos[t] for t in types])
		# rhoH = rhoVec[np.newaxis, np.newaxis, :]*H

		# pl.figure( figsize=(10, 10))
		# pl.title("rho H vs q")
		# for p in H:
		#  	pl.plot(qs, rhos[p[0]]*H[p], label=p)
		# pl.legend()
		
		from matplotlib.colors import hsv_to_rgb
		basisLs = np.geomspace(6, 25, 3) # [2.5, 5, 10, 20, 40, 80]
		# basisLs = [1,2, 3, 6, 12]
		basis = [np.exp(-rs**2/(2*l**2)) for l in basisLs]
		cx = im.getC(dr, n, basis)
		cFromBasis =  im.getCFromBasis(cx, basis, dr, n)
		pl.figure( figsize=(10, 10))
		pl.title("c from basis (-- unconstrained)")
		pl.grid()
		i=0
		for p in cFromBasis:
			col = hsv_to_rgb((i/len(cFromBasis),1,1))
			# pl.plot(rs, cFromBasis[p], label=p, color=col)
			# pl.plot(rs, c[p], linestyle='--', color=col)
			pl.plot(rs, cReg[p], linestyle='-', color=col, label=p)
			i+=1
		# for b in basis:
			# pl.plot(rs, b, linestyle=':', color='gray')
		pl.legend()

		pl.figure( figsize=(10, 10))
		pl.title("c from Basis vs q (-- unconstrained)")
		i=0
		for p in cFromBasis:
			col = hsv_to_rgb((i/len(cFromBasis),1,1))
			# pl.plot(qs, radialFourierTransformRToQ(rs, cFromBasis[p])[1], label=p, color=col)
			pl.plot(qs, radialFourierTransformRToQ(rs, c[p])[1], linestyle='--', color=col)
			pl.plot(qs, radialFourierTransformRToQ(rs, cReg[p])[1], linestyle=':', color=col)
			i+=1
		# for b in basis:
			# pl.plot(qs, radialFourierTransformRToQ(rs, b)[1]/np.sum(b), linestyle=':', color='gray')
		pl.legend()

		# print("cFromBasis", cFromBasis)

		gFromcFromBasis = im.getGFromC(cFromBasis, dr, n)
		gFromcReg = im.getGFromC(cReg, dr, n)
		# print("gFromcFromBasis", gFromcFromBasis)

		pl.figure( figsize=(10, 10))
		pl.title("g from c from Basis (-- unconstrained)")
		i=0
		for p in gFromcFromBasis:
			col = hsv_to_rgb((i/len(cFromBasis),1,1))
			# pl.plot(rs, gFromcFromBasis[p][1], label=p, color=col)
			pl.plot(rs, gFromcReg[p][1], linestyle=':', label=p, color=col)
			pl.plot(rs, g[p], linestyle='--', color=col)
			i+=1
		pl.legend()


		pl.show()
		exit(0)

		# g, c, rHmat, DMat = im.getRadialDistributionFunction(res)
		h, c, rHmat = im.getRadialDistributionFunction(res)
		# print(g, c, rHmat, DMat)

		# pl.figure( figsize=(10, 10))
		# pl.title("g")
		# for p in g:
		# 	if True or p[0]==p[1]:
		# 		pl.plot(g[p][0], g[p][1], label=p)
		# pl.legend()

		pl.figure( figsize=(10, 10))
		pl.title("c")
		for p in c:
			if True or p[0]==p[1]:
				pl.plot(c[p][0], c[p][1], label=p)
		pl.legend()

		pl.figure( figsize=(10, 10))
		pl.title("h")
		for p in h:
			if True or p[0]==p[1]:
				pl.plot(h[p][0], h[p][1], label=p)
		pl.legend()

		pl.figure( figsize=(10, 10))
		pl.title("rHmat")
		for i in range(len(im.cells)):
			for j in range(len(im.cells)):
				pl.plot(qs, rHmat[:,i,j], label=f"{i}, {j}")
		pl.legend()
		
		# pl.figure( figsize=(10, 10))
		# pl.title("D")
		# for i in range(len(im.cells)):
		# 	for j in range(len(im.cells)):
		# 		pl.plot(qs, DMat[:,i,j], label=f"{i}, {j}")
		# pl.legend()

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