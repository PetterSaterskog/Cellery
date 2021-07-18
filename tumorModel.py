import numpy as np
from numpy import newaxis
import matplotlib.pyplot as pl
from scipy.spatial import cKDTree
from scipy.special import gamma, k0, k1
from scipy.signal import convolve
from scipy.interpolate import interpn

types = ['healthy', 'cancer', 'immune']

def sphereVol(d):
	return np.pi**(d/2)/gamma(d/2 + 1)

class TumorModel():
	def __init__(self, cellEffR={t:10 for t in types}, immuneFraction=0, neighborDist=30, growth={'cancer':1.,'immune':0.}, diffusion={}, infectionRange=200.):
		self.cellEffRadVec = np.array([cellEffR[t] for t in types]).astype(np.float32)
		self.immuneFraction = immuneFraction
		self.growth = growth
		self.neighborDist = neighborDist
		self.diffusionMat = np.zeros([len(types)]*2)
		for p in diffusion:
			self.diffusionMat[types.index(p[0]), types.index(p[1])] = diffusion[p]
			self.diffusionMat[types.index(p[1]), types.index(p[0])] = diffusion[p]
		self.infectionRange = infectionRange

class Tumor():
	def __init__(self, tumorModel, d=2, L=1500, tumorCellCount=1500, verbose=False, saveEvolution=False, maxSteps=-1):
		self.L = L
		effVol = (sphereVol(d)*tumorModel.cellEffRadVec**d).dot( [1-tumorModel.immuneFraction, 0, tumorModel.immuneFraction] ) #fraction weighted average volume per cell
		nCells = int(L**d / effVol)
		nImmuneCells = int(nCells*tumorModel.immuneFraction)
		self.cellPositions = (np.random.rand(nCells, d)*L).astype(np.float32)
		self.cellTypes = np.zeros((nCells,), dtype=np.int32)

		self.cellTypes[:nImmuneCells] = 2
		self.cellPositions[nImmuneCells] = L/2
		self.cellTypes[nImmuneCells] = 1

		res = 20
		gridN = int(L/res)
		gridDx = L/gridN
		kernelXs = np.linspace(-gridN*gridDx, gridN*gridDx, 2*gridN+1)
		self.gridEdges = np.linspace(0, L, gridN+1)
		self.gridCenters = (self.gridEdges[:-1] + self.gridEdges[1:])/2
		if d==2:
			kernelXY = np.array([[[x, y] for y in kernelXs] for x in kernelXs])
			kernelRs = np.linalg.norm(kernelXY, axis=2)
			
			self.fieldKernel = k0(kernelRs/tumorModel.infectionRange)
			self.fieldGradientKernel = -(k1(kernelRs/tumorModel.infectionRange)/(kernelRs*tumorModel.infectionRange))[newaxis,:,:] * np.moveaxis(kernelXY,2,0)
			self.fieldKernel[gridN, gridN] = self.fieldKernel[gridN + 1, gridN]
			self.fieldGradientKernel[:, gridN, gridN] = 0
			repStrength = 50*self.fieldKernel[gridN, gridN]
			repDist = 0.5*tumorModel.neighborDist
			self.fieldKernel -= repStrength*np.exp(-(kernelRs/repDist)**2)
			self.fieldGradientKernel -= (2*repStrength*np.exp(-(kernelRs/repDist)**2)/repDist**2)[newaxis,:,:] * np.moveaxis(kernelXY,2,0)
			#TODO finite diff check
		else:
			assert(False)
		
		tumorCells = 1
		dt = 0.2 / max(np.max(list(tumorModel.growth.values())), np.max(tumorModel.diffusionMat), tumorModel.moveSpeed) #ensure the most common event only happens every 5 steps for small dt convergence
		
		if saveEvolution:
			self.evolution = [(self.cellPositions, self.cellTypes)]
		while tumorCells < tumorCellCount and (not saveEvolution or len(self.evolution)<maxSteps):
			tumorCells += self.update(tumorModel, dt)
			if saveEvolution:
				self.evolution.append((self.cellPositions, self.cellTypes))
			if verbose:
				print(f"{tumorCells} tumor cells: {100*np.log(tumorCells)/np.log(tumorCellCount):.1f}%")
				print(f"{np.sum(self.cellTypes == 2)} immune cells")

	def addSources(self, sourcePoints, sourceAreas):
		diffs = self.cellPositions[:, newaxis,:] - sourcePoints[newaxis, :,:]
		ns = np.sum(diffs**2, axis=2)
		ns[ns==0] = 1
		sqrtArg = sourceAreas[newaxis,:]*(1-np.exp(-ns*np.pi/np.abs(sourceAreas)[newaxis,:]))/(ns*np.pi) + 1
		self.cellPositions += np.sum(diffs * (np.sqrt(sqrtArg) - 1)[:,:,newaxis], axis=1)

	def update(self, tumorModel, dt):
		tree = cKDTree(self.cellPositions)
		pairs = tree.query_pairs(tumorModel.neighborDist, output_type="ndarray")

		cancerCounts = np.histogramdd(self.cellPositions[self.cellTypes==1], bins = self.cellPositions.shape[1]*[self.gridEdges])[0]
		self.infection = convolve(cancerCounts, self.fieldKernel, mode='same')
		self.infectionGradient = np.moveaxis([convolve(cancerCounts, kernel, mode='same') for kernel in self.fieldGradientKernel],0,2) #scipy convolve cannot handle higher dim kernels..

		for i in range(len(types)):
			for j in range(len(types)):
				if tumorModel.diffusionMat[i,j]>0:
					allPairs = pairs[(self.cellTypes[pairs[:,0]]==i) & (self.cellTypes[pairs[:,1]]==j), :]
					for p in np.random.permutation(allPairs[np.random.rand(allPairs.shape[0]) < dt * tumorModel.diffusionMat[i,j]]): #permute to avoid finite dt effects being biased based on nn search order
						if self.cellTypes[p[0]]==i and self.cellTypes[p[1]]==j: #don't swap cells that have already been swapped..
							self.cellTypes[p[0]], self.cellTypes[p[1]] = self.cellTypes[p[1]], self.cellTypes[p[0]]


		rad = 10
		diff = self.cellPositions[pairs[:,1]]-self.cellPositions[pairs[:,0]]
		diff2 = np.sum(diff*diff, axis=1)
		w = 5*np.exp(-diff2/rad**2)/rad**2
		f = 10*diff*w[:, newaxis]/np.sqrt(diff2)[:, newaxis]
		np.add.at(self.cellPositions,pairs[:,0], -f)
		np.add.at(self.cellPositions,pairs[:,1], f)

		ccs = np.arange(len(self.cellPositions))[self.cellTypes==1]
		ics = np.concatenate( (pairs[(self.cellTypes[pairs[:,0]]==2) & (self.cellTypes[pairs[:,1]]==1), 0],
							   pairs[(self.cellTypes[pairs[:,1]]==2) & (self.cellTypes[pairs[:,0]]==1), 1]) )

		toSplit = np.concatenate( (ccs[np.random.rand(ccs.shape[0]) < dt*tumorModel.growth['cancer']], ics[np.random.rand(ics.shape[0]) < dt*tumorModel.growth['immune']] ))
		
		self.addSources(self.cellPositions[toSplit,:], tumorModel.cellEffRadVec[self.cellTypes[toSplit]]**2*np.pi )

		if True:
			toMove = np.arange(len(self.cellPositions))[self.cellTypes==2]
			toMove = toMove[np.random.rand(toMove.shape[0]) < dt*tumorModel.moveSpeed]
			oldPositions = np.array(self.cellPositions[toMove])
			grads = interpn(self.cellPositions.shape[1]*[self.gridCenters], self.infectionGradient, self.cellPositions[toMove, :], bounds_error = False, fill_value=None)
			gradNorm = np.linalg.norm(grads, axis=1)[:, newaxis]
			gradNorm[gradNorm==0]=1
			moveVecs = 15*grads / gradNorm
			self.cellPositions[toMove] += moveVecs

			self.addSources(np.concatenate([oldPositions, self.cellPositions[toMove,:]]), np.concatenate([-tumorModel.cellEffRadVec[self.cellTypes[toMove]]**2*np.pi, tumorModel.cellEffRadVec[self.cellTypes[toMove]]**2*np.pi]) )
		# self.addSources(self.cellPositions[toMove,:], tumorModel.cellEffRadVec[self.cellTypes[toMove]]**2*np.pi )

		# diffs = self.cellPositions[:, newaxis,:] - self.cellPositions[newaxis, toSplit,:]
		# ns = np.sum(diffs**2, axis=2)
		# ns[ns==0] = 1
		# r2s = tumorModel.cellEffRadVec[self.cellTypes[toSplit]]**2
		# self.cellPositions += np.sum(diffs * (np.sqrt(r2s[newaxis,:]/ns + 1) - 1)[:,:,newaxis], axis=1)

		randDirs = np.random.normal(size=(len(toSplit),2)).astype(np.float32)
		randDirs = tumorModel.cellEffRadVec[self.cellTypes[toSplit]][:, newaxis]*randDirs/np.linalg.norm(randDirs, axis=1)[:,newaxis]
		self.cellPositions = np.concatenate((self.cellPositions, self.cellPositions[toSplit] + randDirs), axis=0)
		self.cellPositions[toSplit] -= randDirs
		self.cellTypes = np.concatenate((self.cellTypes, self.cellTypes[toSplit]), axis=0)

		inside = np.all((self.cellPositions > 0) & (self.cellPositions < self.L), axis=1)
		self.cellPositions, self.cellTypes = self.cellPositions[inside], self.cellTypes[inside]
		
		return toSplit.shape[0]

def plot(cellPositions, cellTypes):
	colors = {'healthy':(0,.8,0), 'cancer':(1,0,0), 'immune':(0,0,1)}
	fig = pl.figure(figsize=(16,16))
	sc=[]
	for i in range(len(types)):
		ps = cellPositions[cellTypes==i]
		sc.append(pl.scatter(ps[:,0], ps[:,1], s=10, color=colors[types[i]], label=types[i]))
	pl.xlabel("x [μm]")
	pl.ylabel("y [μm]")
	pl.axis('square')
	pl.legend(loc="upper right")
	return fig, sc
	
if __name__ == "__main__":
	tm = TumorModel(immuneFraction=0.03, growth={'cancer':0.3, 'immune':0*0.1}, diffusion = {('cancer', 'immune'):0.0, ('cancer', 'healthy'):0.4, ('healthy','immune'):1.5})
	tm.moveSpeed = 2.1
	tumor = Tumor(tm, verbose=True, saveEvolution=True, L=4000,tumorCellCount=3000, maxSteps=600)
	
	fig, sc = plot(tumor.cellPositions, tumor.cellTypes)

	def updateFig(frameI):
		for i in range(len(types)):
			sc[i].set_offsets(tumor.evolution[frameI][0][tumor.evolution[frameI][1]==i])
	import matplotlib.animation as animation
	anim = animation.FuncAnimation(fig, updateFig, frames=len(tumor.evolution))
	writervideo = animation.FFMpegWriter(fps=30) 
	anim.save(f"out/test.avi", writer=writervideo)


	pl.figure()
	pl.imshow(tumor.infection[:,:])
	pl.figure()
	pl.imshow(tumor.infectionGradient[:,:,0])
	pl.figure()
	pl.imshow(tumor.infectionGradient[:,:,1])
	pl.show()
