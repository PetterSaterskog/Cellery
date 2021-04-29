import numpy as np
import matplotlib.pyplot as pl
import os
from tqdm import tqdm
from collections import defaultdict
import matplotlib.patches as patches
from multiprocessing import Pool

inputDir = "cell_positions_data"
figDir = "figs/"
patientList = ['patient13','patient23','patient36', 'patient39']
patientList = ['patient39']

cells = []
types = [{} for i in range(50)]

def typeId(pi,t):
	if not t in types[pi]:
		types[pi][t] = len(types[pi])
	return types[pi][t] 

margin = 15
L = 800 - 2*margin
names = []
for f in os.listdir(inputDir):
	name = f[:-19]
	if name in patientList: #['patient23']:
		names.append(name)
		print("{}: ".format(len(cells))+f)
		rows = np.loadtxt(inputDir+'/'+f, delimiter=',', skiprows=1, converters = {3: lambda s: typeId(len(cells), s.decode("utf-8"))})
		inside = (rows[:,0]>margin)*(rows[:,0] < L+margin)*(rows[:,1]>margin)*(rows[:,1] < L+margin)
		print('{:.1f}% inside margins.'.format(100*np.sum(inside)/rows.shape[0]))
		cells.append(rows[ np.where(inside) ]) #filter out everything outside margin

cellsByType = []

for pi,p in enumerate(cells):
	d = [[] for i in types[pi]]
	for c in p:
		d[int(c[3])].append(c[:2] - margin)
	for i in range(len(types[pi])):
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
typeOrder = [*cancerCellTypes, *otherCellTypes, *TCellTypes, *BCellTypes, *otherImmuneCellTypes, *unknownCellTypes]

cellMarkers = {'CD8-T':('o', (0.9,0.4,0.6)),
			   'CD4-T':('o', (0.2,1.0,0.2)),
			   'CD3-T':('o', (0.4,0.4,0.9)),
			   'Tregs':('o', (0.9,0.4,0.9)),
			   'B':('$\mathrm{Y}$', (0.9,0.4,0.6)),
			   'Macrophages':('^', (0.3,0.8,0.3)),
			   'Other immune':('^', (0.5,0.5,1.0)),
			   'Mono / Neu':('^', (0.9,0.9,0.0)),
			   'DC / Mono':('^', 'royalblue'),
			   'Neutrophils':('^', 'forestgreen'),
			   'DC':('^', 'darkviolet'),
			   'NK':('^', 'crimson'),
			   'Kreatin-positive tumor':('*', 'orange'),
			   'Tumor':('*','red'),
			   'Endothelial':('P','maroon'),
			   'Mesenchymal-like':('P','gray'),
			   'Unidentified':('$?$','gray')
			   }

# edges = np.sqrt(np.linspace(0, 100**2, 100))
# edges = np.linspace(0, 100, 60*2)
edges = np.linspace(0, 200, 60*2)
# edgeEffect = getCorr(np.random.rand(4*10**4, 2)*L, np.random.rand(4*10**4, 2)*L, edges)

#doesn't sample pairs between a cell and itself if c2 is c1
def getCorr(c1, c2, edges, L, same):
	areas = np.diff(np.pi*edges**2)
	if same: #c2 is c1:
		pairs = cKDTree(c1).query_pairs(edges[-1], output_type =  'ndarray')
		norm1 = 2 / areas / len(c1) / (len(c1)-1) * L**2
	else:
		neighbors = cKDTree(c1).query_ball_tree(cKDTree(c2), edges[-1])
		pairs = np.array([(i, j) for i in range(len(c1)) for j in neighbors[i]])
		norm1 = 1 / areas / len(c1) / len(c2) * L**2
	if len(pairs)==0:
		dists = []
	else:
		dists = np.linalg.norm(c2[pairs[:,1]] - c1[pairs[:,0]], axis=1)
	hist = np.histogram(dists, edges)[0]
	return hist * norm1, np.sqrt(hist + 1) * norm1

def plotHist(binEdges, counts, ls='-', label=''):
	pl.plot(np.repeat(binEdges, 2)[1:-1], np.repeat(counts, 2), ls, label=label)

#assumes equal sized bins starting at 0
def gContrib(ps, p, edges, norm2):
	# return np.histogram(np.linalg.norm(p[np.newaxis, :] - ps, axis = 1), bins = (len(edges)-1), range=(0, edges[-1]) )[0] * norm
	return np.histogram(np.linalg.norm(p[np.newaxis, :] - ps, axis = 1), bins = edges )[0] * norm2

def update(ps, ucorr, g, gErr, chi2, sigma, edges, L, ratios, norms):
	ti = np.random.choice(len(ps), p=ratios)
	i = np.random.randint(len(ps[ti]))
	newPos = np.random.rand(2) * L #np.remainder(ps[ti][i] + np.random.normal()*xSigma, L)

	newCorr = {}
	for oti in range(len(ps)):
		pair = (min(ti, oti), max(ti, oti))
		if oti==ti:
			newCorr[pair] = ucorr[pair] - gContrib(ps[ti][:i], ps[ti][i], edges, norms[pair]) - gContrib(ps[ti][i+1:], ps[ti][i], edges, norms[pair]) + gContrib(ps[ti][:i], newPos, edges, norms[pair]) + gContrib(ps[ti][i+1:], newPos, edges, norms[pair])
		else:
			newCorr[pair] = ucorr[pair] - gContrib(ps[oti], ps[ti][i], edges, norms[pair]) + gContrib(ps[oti], newPos, edges, norms[pair])



	def getPartChi2(c1, c2):
		chi2 = 0
		for oti in range(len(ps)):
			pair = (min(ti, oti), max(ti, oti))
			chi2 += np.sum(((c1[pair] - c2[pair])/gErr[pair])**2)
		return chi2

	newChi2 = chi2 - getPartChi2(ucorr, g) + getPartChi2(newCorr, g)

	if newChi2 < chi2 or newChi2-chi2 < (np.random.normal()*sigma)**2:
		ps[ti][i] = newPos
		chi2 = newChi2
		for oti in range(len(ps)):
			pair = (min(ti, oti), max(ti, oti))
			
			# ref = getTypeCorrelators(ps, edges, L)
			# print(newCorr[pair] - corr[pair][:])
			# print(ref[pair] - corr[pair][:])

			ucorr[pair][:] = newCorr[pair]
			# assert(False)
			# assert(np.linalg.norm(corr[pair] - ref[pair]) < 1e-6)
		return 1, chi2
	return 0, chi2

def getTypeCorrelators(cs, edges, L):
	corrAndErr = {(ti,tj):getCorr(cs[ti], cs[tj], edges, L, ti==tj) for ti in range(len(cs)) for tj in range(len(cs)) if ti<=tj}
	return {k:v[0] for k,v in corrAndErr.items()}, {k:v[1] for k,v in corrAndErr.items()}

def processPatient(i):
	verbose = i==0
	cbt = cellsByType[i]
	pos = cells[i][:,:2] - margin
	ratios = [len(cs) for cs in cbt]
	ratios /= np.sum(ratios)

	# for ii in tqdm(range(len(pos))):
	# 	for jj in range(len(pos)):
	# 		if ii!=jj and np.linalg.norm(pos[ii] - pos[jj])==0:
	# 			print(pos[ii])

	g0, g0Err = getTypeCorrelators(cbt, edges, L)
	gPos = [np.random.rand(len(cbt[ti]), 2)*L for ti in range(len(cbt))] # pos
	sigma = 0.
	corr,_ = getTypeCorrelators(gPos, edges, L)

	chi2 = sum( np.sum(((corr[(ti,tj)] - g0[(ti,tj)]) / g0Err[(ti,tj)])**2) for ti in range(len(cbt))  for tj in range(len(cbt)) if ti<=tj)

	nIts = 50_00 #10_000_000
	
	areas = np.diff(np.pi*edges**2)
	norms = {p:((2  / len(cbt[p[0]]) / (len(cbt[p[0]]) - 1)) if p[0]==p[1] else (1 / len(cbt[p[0]]) / len(cbt[p[1]]))) * L**2 / areas for p in corr}
	nBatches = 10
	# xSigma = 100
	
	for bi in range(nBatches):
		accepts = 0
		if verbose: print(f"\nBatch {bi+1}/{nBatches}")
		for j in range(nIts):
			accepted, chi2 = update(gPos, corr, g0, g0Err, chi2, sigma, edges, L, ratios, norms)
			accepts += accepted
		ar = accepts/nIts
		if verbose: print(f'Chi2: {chi2:.4g}')
		if verbose: print(f'Acceptance ratio: {ar}')
		# if ar<0.3:
			# xSigma /= 2
	return i, cbt, gPos, g0, corr

def plotPatient(i, cbt, gPos, g0, corr):
	finalCorr,_ = getTypeCorrelators(gPos, edges, L)

	# for p in g0:
	# 	pl.figure(figsize=(5, 5))
	# 	plotHist(edges, g0[p], ls='-', label='original')
	# 	plotHist(edges, corr[p], ls='-', label='gen acc')
	# 	plotHist(edges, finalCorr[p], ls='--', label='gen scratch')
	# 	pl.legend()
	# 	pl.savefig(figDir+f"corr_{p}_{names[i]}.pdf")
	# 	pl.close()

	# pl.figure(figsize=(15, 15))
	# pl.plot(gPos[:,0], gPos[:,1],linestyle="None", marker=".")
	# pl.savefig(figDir+f"allCellsGen_{names[i]}.pdf")
	# pl.close()
	
	# pl.figure(figsize=(15, 15))
	# pl.plot(pos[:,0], pos[:,1],linestyle="None", marker=".")
	# pl.savefig(figDir+f"allCells_{names[i]}.pdf")
	# pl.close()

	vs = {'g':gPos, 'o':cbt}
	for v in vs:
		tot =  sum(len(t) for t in vs[v])
		pl.figure(figsize=(15, 15))
		# pl.gca().set_facecolor((0.5, 0.5, 0.5))
		for typeName in typeOrder:
			if not typeName in types[i]: continue
			pCells = vs[v][typeId(i, typeName)]
			pl.plot(pCells[:,0],
					pCells[:,1],
					linestyle="None",
					marker=cellMarkers[typeName][0],
					color=cellMarkers[typeName][1],
					label = f"{typeName} ({100*len(pCells)/tot:.1f}%)")

		pl.gca().add_patch(patches.Rectangle((0, 0), L, L,  linewidth=1, edgecolor='r', facecolor='none'))
		pl.legend(loc = 'right')
		pl.xlim((0, L*1.3))
		pl.ylim((0, L*1.3))
		pl.title(f'{names[i]} - {tot} cells')
		pl.savefig(figDir+f"cellsByType_{v}_{names[i]}.pdf")
		pl.close()

with Pool(50) as pool:
	res = pool.map(processPatient, range(len(cellsByType)))
	for r in res:
		plotPatient(*r)