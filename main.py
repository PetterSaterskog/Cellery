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

patientList = ['patient1','patient2','patient13','patient14','patient23','patient36','patient37', 'patient39']
# patientList = ['patient1','patient2','patient14','patient23','patient36','patient37', 'patient39']
# patientList = ['patient13','patient14','patient23','patient36','patient39']
# patientList = ['patient39']

images = {}
for fn in os.listdir(inputDir):
	name = fn[:-19]
	if name in patientList:
		print(f"{name}:")
		im = keren.loadImage(fn)
		counts = im.getCounts()
		images[name] = im

def getAverageS0(qs):
	

edges = np.linspace(0, 300, 100)

for fn in os.listdir(inputDir):
	name = fn[:-19]
	if name in patientList:
		print(f"{name}:")
		im = keren.loadImage(fn)
		counts = im.getCounts()
		images[name] = im
		# corrs, err = images[name].getCorrs(edges)
		# pl.figure()
		# for c in corrs:
		# 	if 'Kreatin-positive tumor' in c:
		# 		plotHist(edges, corrs[c])
		# pl.show()
		resolution = 3 #um
		qs = np.linspace((2*np.pi)/(0.9*im.W/2), 2*np.pi / resolution, 100)
		S0 = images[name].getStructureFactors(qs)[0]
		plotCells(im.cells, im.W, f"b_{name}", f"b_{name}")
		# for i in range(500):
		import time
		i=[0]

		def callback(cd):
			plotCells(cd.cells, cd.W, f"a_{name}, iteration={i[0]}", f"{name}")
			i[0]+=1

		#start = time.time()
		resolutions = [60, 55, 50, 45, 40, 35, 30, 25, 20, 12, 6, 3] #um
		step = 5.
		cells = {c:np.random.rand(counts[c], 2)*im.W for c in counts}
		guess = CellDistribution(cells, im.W, torus=True)
		for resolution in resolutions:
			minQ = (2*np.pi)/(0.9*im.W/2)
			maxQ = 2*np.pi/resolution
			qs = np.linspace(minQ, maxQ, max(2, int(2*maxQ//minQ)))
			print(f"{len(qs)} wavevectors:")
			S0 = images[name].getStructureFactors(qs)[0]
			L0, = guess.getSLoss(S0, qs)
			print(f"optimizing..")
			gim, step = generateCellDistribution({t:counts[t]//1 for t in counts}, S0, im.W//1, qs, start = cells, maxIterations = 15,  callback=callback, startStep = step, L0=L0)
			cells = gim.cells

		with open(f'{name}.pickle', 'wb') as h:
			pickle.dump((im.W, cells), h)
		
		#end = time.time()

		#print("Time: ", end - start)
		# plotCells(gim.cells, gim.W, f"a_{name}, iteration={i}")

	#getCellDistribution(densities, correlators, nIterations)

exit(0)


# def processPatient(i):
# 	verbose = i==0
# 	cbt = cellsByType[i]
# 	pos = cells[i][:,:2] - margin
# 	ratios = [len(cs) for cs in cbt]
# 	ratios /= np.sum(ratios)

# 	# for i in range(len(cbt)):
# 		# print(f'{i}: {len(cbt[i])}')
# 	# for ii in tqdm(range(len(pos))):
# 	# 	for jj in range(len(pos)):
# 	# 		if ii!=jj and np.linalg.norm(pos[ii] - pos[jj])==0:
# 	# 			print(pos[ii])

# 	edges = np.linspace(0, 300, 60*2)
# 	g0, g0Err = getTypeCorrelators(cbt, edges, L)

# 	if False:
# 		edges3 = np.linspace(0, np.sqrt(200), 20)**2
# 		edges3 = np.linspace(0, 100, 40)
# 		g3, g3Err = getTypeCorrelators3(cbt, edges3, L)

# 		pt=2
# 		for i in range(0, len(edges3) - 1, 3):
# 			pl.figure()
# 			pl.title(f"d_1={(edges3[i]+edges3[i+1])/2:.1f} um")
# 			pl.imshow(g3[(pt,pt,pt)][i,:,:], origin='lower')
# 			pl.xlabel("d_2 [um]")
# 			pl.ylabel("d_3 [um]")
# 		pl.show()
# 		exit()

# 	gPos = [np.random.rand(len(cbt[ti]), 2)*L for ti in range(len(cbt))] # pos
# 	sigma = 0.
# 	corr,_ = getTypeCorrelators(gPos, edges, L)

# 	chi2 = sum( np.sum(((corr[(ti,tj)] - g0[(ti,tj)]) / g0Err[(ti,tj)])**2) for ti in range(len(cbt))  for tj in range(len(cbt)) if ti<=tj)

# 	nIts = 5_000*10_00 #10_000_000
	
# 	areas = np.diff(np.pi*edges**2)
	
# 	norms = {p:((2  / len(cbt[p[0]]) / (len(cbt[p[0]]) - 1)) if p[0]==p[1] else (1 / len(cbt[p[0]]) / len(cbt[p[1]]))) * L**2 / areas for p in corr if (p[0]!=p[1] and len(cbt[p[0]]) and len(cbt[p[1]])) or len(cbt[p[0]])>1}
	
# 	getCellDistribution()

# 		# if ar<0.3:
# 			# xSigma /= 2
# 	return i, cbt, gPos, g0, corr, edges

# def plotPatient(i, cbt, gPos, g0, corr, edges):
# 	finalCorr,_ = getTypeCorrelators(gPos, edges, L)

# 	for p in g0:
# 		pl.figure(figsize=(5, 5))
# 		plotHist(edges, g0[p], ls='-', label='original')
# 		plotHist(edges, corr[p], ls='-', label='gen acc')
# 		plotHist(edges, finalCorr[p], ls='--', label='gen scratch')
# 		pl.legend()
# 		pl.savefig(figDir+f"corr/{p}_{names[i]}.pdf")
# 		pl.close()

# 	vs = {'g':gPos, 'o':cbt}
# 	for v in vs:
		
# 		plotCells(f'{v}_{names[i]}')

# if len(patientList)==1:
# 	plotPatient(*processPatient(0))
# else:
# 	with Pool(50) as pool:
# 		res = pool.map(processPatient, range(len(cellsByType)))
# 		for r in res:
# 			plotPatient(*r)
