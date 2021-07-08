import numpy as np
import matplotlib.pyplot as pl
import scipy
from scipy import signal
import matplotlib.animation as animation
from scipy.spatial import cKDTree
import time


#micrometers
L = 5000
d = 2
dt = 0.1 #hours
diffusion = {('cancer', 'immune'):0.0,
				('cancer', 'healthy'):0.5,
				('healthy','immune'):1.5}

cellRad = 10 #this assumes no gap, just defined through volPerCell
neighborDist = cellRad * 3
volPerCell = cellRad**2*np.pi
immuneFraction = 0.03
nCancer = 1

nCells = int(L**2 / volPerCell)

types = ['healthy', 'cancer', 'immune']
colors = {'healthy':(0,1,0), 'cancer':(1,0,0), 'immune':(0,0,1)}

diffusionMat = np.zeros([len(types)]*2)
for p in diffusion:
	diffusionMat[types.index(p[0]), types.index(p[1])] = diffusion[p]
	diffusionMat[types.index(p[1]), types.index(p[0])] = diffusion[p]

cellPositions = np.random.rand(nCells, d)*L
cellTypes = np.zeros((nCells,), dtype=np.int32)

cellTypes[:int(nCells*immuneFraction)] = 2
cellPositions[-nCancer:] = (np.random.rand(nCancer, d)-.5)*500+L/2
cellTypes[-nCancer:] = 1
# cellPositions[-1, :] = np.ones((2,))*L/2

cancerGrowth = 0.03
immuneGrowth = 0.01

frames = 0
def updatefig(*args):

	start = time.time()

	global frames, cellPositions, cellTypes
	
	tree = cKDTree(cellPositions)
	end = time.time()
	print(f"Create tree: {end - start:.2f} s")
	start = end

	pairs = tree.query_pairs(neighborDist, output_type="ndarray")
	end = time.time()
	print(f"Find pairs: {end - start:.2f} s")
	start = end

	newCells = []

	displacements = np.zeros(cellPositions.shape)

	for i in range(len(types)):
		for j in range(len(types)):
			if diffusionMat[i,j]>0:
				allPairs = pairs[(cellTypes[pairs[:,0]]==i) & (cellTypes[pairs[:,1]]==j), :]
				for p in np.random.permutation(allPairs[np.random.rand(allPairs.shape[0]) < dt * diffusionMat[i,j]]):
					if cellTypes[p[0]]==i and cellTypes[p[1]]==j: #hasn't already been swapped..
						cellTypes[p[0]], cellTypes[p[1]] = cellTypes[p[1]], cellTypes[p[0]]

	end = time.time()
	print(f"Swap cells: {end - start:.2f} s")
	start = end

	splitRad = cellRad * 1.5

	ccs = np.arange(len(cellPositions))[cellTypes==1]
	ics = np.concatenate( (pairs[(cellTypes[pairs[:,0]]==2) & (cellTypes[pairs[:,1]]==1), 0], pairs[(cellTypes[pairs[:,1]]==2) & (cellTypes[pairs[:,0]]==1), 1]) )

	toSplit = np.concatenate( (ccs[np.random.rand(ccs.shape[0]) < dt*cancerGrowth], ics[np.random.rand(ics.shape[0]) < dt*immuneGrowth] ))

	diffs = cellPositions[:, np.newaxis,:] - cellPositions[np.newaxis, toSplit,:]
	ns = np.sum(diffs**2, axis=2)
	ns[ns==0] = 1
	cellPositions += np.sum(diffs * (np.sqrt(cellRad**2/ns + 1) - 1)[:,:,np.newaxis], axis=1)
	
	end = time.time()
	print(f"Split cells: {end - start:.2f} s")
	start = end

	randDirs = np.random.normal(size=(len(toSplit),2))
	randDirs = 0.5*splitRad*randDirs/np.linalg.norm(randDirs, axis=1)[:, np.newaxis]
	cellPositions = np.concatenate((cellPositions, cellPositions[toSplit] + randDirs), axis=0)
	cellPositions[toSplit] -= randDirs
	cellTypes = np.concatenate((cellTypes, cellTypes[toSplit]), axis=0)

	#remove cells outside
	inside = np.all((cellPositions > 0) & (cellPositions < L), axis=1)
	cellPositions, cellTypes = cellPositions[inside], cellTypes[inside]

	end = time.time()
	print(f"Remove outside: {end - start:.2f} s")
	start = end
	
	for i in range(len(types)):
		sc[i].set_offsets(cellPositions[cellTypes==i])
	s=""
	for i in range(len(types)):
		s+=f"{i}: {np.sum(cellTypes==i)}, "

	
	print(f"frame {frames}: {s[:-2]}")
	frames += 1

	end = time.time()
	print(f"Rest: {end - start:.2f} s")
	start = end
	return sc

fig = pl.figure(figsize=(16,16))
sc = []
for i in range(len(types)):
	ps = cellPositions[cellTypes==i]
	sc.append( pl.scatter(ps[:,0],ps[:,1], s=2, color=colors[types[i]]) )
pl.xlabel("x [μm]")
pl.ylabel("y [μm]")

anim = animation.FuncAnimation(fig, updatefig, interval=10, blit=True, frames=100_000)
writervideo = animation.FFMpegWriter(fps=30) 

outFolder = "out/tumorSim"
from pathlib import Path
Path(outFolder).mkdir(parents=True, exist_ok=True)
diff=str(list(diffusion.values()))
anim.save(f"{outFolder}/tumor_L={L}_diffusion={diff[1:-1]}_cancerGrowth={cancerGrowth}_immuneGrowth={immuneGrowth}.avi", writer=writervideo)

# pl.show()
