import numpy as np
import matplotlib.pyplot as pl
from scipy.spatial import cKDTree
from scipy.special import gamma

types = ['healthy', 'cancer', 'immune']

def SphereVol(d):
	return np.pi**(d/2)/gamma(d/2 + 1)

class TumorModel():
	def __init__(self, cellEffR={t:10 for t in types}, immuneFraction=0, neighborDist=30, growth={'cancer':1.,'immune':0.}, diffusion={}):
		self.cellEffRadVec = np.array([cellEffR[t] for t in types])
		self.immuneFraction = immuneFraction
		self.growth = growth
		self.neighborDist = neighborDist
		self.diffusionMat = np.zeros([len(types)]*2)
		for p in diffusion:
			self.diffusionMat[types.index(p[0]), types.index(p[1])] = diffusion[p]
			self.diffusionMat[types.index(p[1]), types.index(p[0])] = diffusion[p]

class Tumor():
	def __init__(self, tumorModel, d=2, L=2000, tumorCellCount=100, verbose=False):
		self.L = L
		effVol = (SphereVol(d)*tumorModel.cellEffRadVec**d).dot( [1-tumorModel.immuneFraction, 0, tumorModel.immuneFraction] ) #fraction weighted average volume per cell
		nCells = int(L**d / effVol)
		nImmuneCells = int(nCells*tumorModel.immuneFraction)
		self.cellPositions = np.random.rand(nCells, d)*L
		self.cellTypes = np.zeros((nCells,), dtype=np.int32)

		self.cellTypes[:nImmuneCells] = 2
		self.cellPositions[nImmuneCells] = L/2
		self.cellTypes[nImmuneCells] = 1
		
		tumorCells = 1
		dt = 0.2 / max(np.max(list(tumorModel.growth.values())), np.max(tumorModel.diffusionMat)) #ensure the most common event only happens every 5 steps for small dt convergence
		while tumorCells < tumorCellCount:
			tumorCells += self.update(tumorModel, dt)
			if verbose:
				print(f"{tumorCells} tumor cells: {100*np.log(tumorCells)/np.log(nCells*tumorCellRatio):.1f}%")

	def update(self, tumorModel, dt):
		tree = cKDTree(self.cellPositions)
		pairs = tree.query_pairs(tumorModel.neighborDist, output_type="ndarray")
		
		newCells = []
		displacements = np.zeros(self.cellPositions.shape)

		for i in range(len(types)):
			for j in range(len(types)):
				if tumorModel.diffusionMat[i,j]>0:
					allPairs = pairs[(self.cellTypes[pairs[:,0]]==i) & (self.cellTypes[pairs[:,1]]==j), :]
					for p in np.random.permutation(allPairs[np.random.rand(allPairs.shape[0]) < dt * tumorModel.diffusionMat[i,j]]): #permute to avoid finite dt effects being biased based on nn search order
						if self.cellTypes[p[0]]==i and self.cellTypes[p[1]]==j: #don't swap cells that have already been swapped..
							self.cellTypes[p[0]], self.cellTypes[p[1]] = self.cellTypes[p[1]], self.cellTypes[p[0]]

		ccs = np.arange(len(self.cellPositions))[self.cellTypes==1]
		ics = np.concatenate( (pairs[(self.cellTypes[pairs[:,0]]==2) & (self.cellTypes[pairs[:,1]]==1), 0],
							   pairs[(self.cellTypes[pairs[:,1]]==2) & (self.cellTypes[pairs[:,0]]==1), 1]) )

		toSplit = np.concatenate( (ccs[np.random.rand(ccs.shape[0]) < dt*tumorModel.growth['cancer']], ics[np.random.rand(ics.shape[0]) < dt*tumorModel.growth['immune']] ))

		diffs = self.cellPositions[:, np.newaxis,:] - self.cellPositions[np.newaxis, toSplit,:]
		ns = np.sum(diffs**2, axis=2)
		ns[ns==0] = 1
		self.cellPositions += np.sum(diffs * (np.sqrt((tumorModel.cellEffRadVec[self.cellTypes[toSplit]]**2)[np.newaxis,:]/ns + 1) - 1)[:,:,np.newaxis], axis=1)

		randDirs = np.random.normal(size=(len(toSplit),2))
		randDirs = tumorModel.cellEffRadVec[self.cellTypes[toSplit]][:, np.newaxis]*randDirs/np.linalg.norm(randDirs, axis=1)[:, np.newaxis]
		self.cellPositions = np.concatenate((self.cellPositions, self.cellPositions[toSplit] + randDirs), axis=0)
		self.cellPositions[toSplit] -= randDirs
		self.cellTypes = np.concatenate((self.cellTypes, self.cellTypes[toSplit]), axis=0)

		inside = np.all((self.cellPositions > 0) & (self.cellPositions < self.L), axis=1)
		self.cellPositions, self.cellTypes = self.cellPositions[inside], self.cellTypes[inside]
		return toSplit.shape[0]

def plot(cellPositions, cellTypes):
	colors = {'healthy':(0,.8,0), 'cancer':(1,0,0), 'immune':(0,0,1)}
	fig = pl.figure(figsize=(16,16))
	for i in range(len(types)):
		ps = cellPositions[cellTypes==i]
		pl.scatter(ps[:,0], ps[:,1], s=1, color=colors[types[i]], label=types[i])
	pl.xlabel("x [μm]")
	pl.ylabel("y [μm]")
	pl.axis('square')
	pl.legend(loc="upper right")
	
if __name__ == "__main__":
	tm = TumorModel(immuneFraction=0.03, growth={'cancer':0.3, 'immune':0.1}, diffusion = {('cancer', 'immune'):0.0, ('cancer', 'healthy'):0.4, ('healthy','immune'):1.5})
	tumor = Tumor(tm, verbose=True)
	plot(tumor.cellPositions, tumor.cellTypes)
	pl.show()
