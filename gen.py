from numba import jit
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as pl
from scipy.spatial import cKDTree

def periodDist(p1, p2, L):
	# d = np.abs(p2 - p1)
	# d= p2-p1
	# return np.sqrt(d.dot(d))
	return np.linalg.norm( p2[np.newaxis,:] - p1 , axis=1) #np.linalg.norm(np.fmin(d, L-d))

def update(cells, nCells, mu, Vs, L):
	change = np.random.randint(3)
	if nCells==0 or change==0:
		#add cell
		pos = L * np.random.rand(2)
		dE = mu
		dE += np.sum(Vs( periodDist(cells[:nCells, :], pos, L) ))
		if np.log(np.random.rand()) < -dE:
			assert(cells.shape[0] > nCells)
			cells[nCells] = pos
			nCells += 1
	elif change==1:
		#remove cell
		i = np.random.randint(nCells)
		dE = -mu
		dE -= np.sum(Vs( periodDist(cells[:i], cells[i], L) )) + np.sum( Vs( periodDist(cells[i+1:nCells], cells[i], L) ))
		if np.log(np.random.rand()) < -dE:
			nCells -= 1
			cells[i] = cells[nCells]
	else:
		#move cell
		i = np.random.randint(nCells)
		dE = -mu
		sigma = 5
		pos = cells[i] + np.random.normal(2)*sigma
		dE -= np.sum(Vs( periodDist(cells[:i], cells[i], L) )) + np.sum( Vs( periodDist(cells[i+1:nCells], cells[i], L) ))
		dE += np.sum(Vs( periodDist(cells[:i], pos, L) )) + np.sum( Vs( periodDist(cells[i+1:nCells], pos, L) ))
		if np.log(np.random.rand()) < -dE:
			cells[i] = pos
		# 	print("moved")
		# else:
		# 	print("didn't move")
	return nCells

	
def V(r):
	# assert(0<r)
	return 5000/r**3

def getCorr(c1, c2, edges):
	areas = np.diff(2*np.pi*edges**2)
	neighbors = cKDTree(c1).query_ball_tree(cKDTree(c2), edges[-1])
	pairs = np.array([(i, j) for i in range(len(c1)) for j in neighbors[i]])
	if len(pairs)==0: return np.zeros((len(edges)-1,))
	dists = np.linalg.norm(c2[pairs[:,1]] - c1[pairs[:,0]], axis=1)
	return np.histogram(dists, edges)[0] / areas / len(c1) / len(c2)

L=4000

maxCells = 10_000
nCells = 0
cells = np.zeros((maxCells, 2))
for j in range(10):
	for i in tqdm(range(40000)):
		nCells = update(cells, nCells, -1, V, L)
	print(nCells)

cells = cells[:nCells, :]

pl.plot(cells[:,0], cells[:,1], linestyle="None", marker=".")


edges = np.linspace(1e-6, 200, 60)
edgeCenters = (edges[:-1] + edges[1:])/2
edgeEffect = getCorr(np.random.rand(4*10**4, 2)*L, np.random.rand(4*10**4, 2)*L, edges)

pl.figure()
hist = getCorr(cells, cells, edges)
pl.plot(edgeCenters, hist)
pl.plot(edgeCenters, edgeEffect)
# pl.ylim([0,2])
pl.xlabel('$r$ [μm]')
pl.ylabel('$f(r)$')

pl.show()