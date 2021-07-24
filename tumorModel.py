import numpy as np
from numpy import newaxis
import matplotlib.pyplot as pl
from scipy.spatial import cKDTree
from scipy.special import gamma, k0, k1
from scipy.signal import fftconvolve
from scipy.interpolate import interpn
from scipy import fft
from sklearn.neighbors import KDTree
from sklearn.neighbors import NearestNeighbors

import time
types = ['healthy', 'cancer', 'immune']


fields = ['']

def sphereVol(d):
	return np.pi**(d/2)/gamma(d/2 + 1)

class TumorModel():
	def __init__(self, cellEffR={t:10 for t in types}, immuneFraction=0, moveSpeed=0, neighborDist=30, growth={'cancer':1.,'immune':0.}, diffusion={}, infectionRange=200.):
		self.cellEffRadVec = np.array([cellEffR[t] for t in types]).astype(np.float32)
		self.immuneFraction = immuneFraction
		self.growth = growth
		self.neighborDist = neighborDist
		self.diffusionMat = np.zeros([len(types)]*2)
		self.moveSpeed = moveSpeed
		for p in diffusion:
			self.diffusionMat[types.index(p[0]), types.index(p[1])] = diffusion[p]
			self.diffusionMat[types.index(p[1]), types.index(p[0])] = diffusion[p]
		self.infectionRange = infectionRange

class Tumor():
	def __init__(self, tumorModel, d=2, L=1500, tumorCellCount=1500, verbose=False, saveEvolution=False, maxSteps=-1, width=20):
		self.L = L
		self.d = d
		effVol = (sphereVol(d)*tumorModel.cellEffRadVec**d).dot( [1-tumorModel.immuneFraction, 0, tumorModel.immuneFraction] ) #fraction weighted average volume per cell
		nCells = int(L**d / effVol)
		nImmuneCells = int(nCells*tumorModel.immuneFraction)
		np.random.seed(0)
		self.cellPositions = (np.random.rand(nCells, d)*L).astype(np.float32)
		self.cellTypes = np.zeros((nCells,), dtype=np.int32)

		self.cellTypes[:nImmuneCells] = 2
		self.cellPositions[nImmuneCells] = L/2
		self.cellTypes[nImmuneCells] = 1

		res = 7
		gridN = int(L/res)
		gridDx = L/gridN
		kernelXs = np.linspace(-gridN*gridDx, gridN*gridDx, 2*gridN+1)
		self.gridEdges = np.linspace(0, L, gridN+1)
		self.gridCenters = (self.gridEdges[:-1] + self.gridEdges[1:])/2
		
		kernelCoords = np.moveaxis(np.meshgrid(*d*[kernelXs], indexing='ij'), 0, d)
		kernelRs = np.linalg.norm(kernelCoords, axis=d)
		kernelRs[gridN, gridN] = 1 #need to treat center specially
		
		self.fieldKernel = k0(kernelRs/tumorModel.infectionRange)
		self.fieldGradientKernel = -(k1(kernelRs/tumorModel.infectionRange)/(kernelRs*tumorModel.infectionRange))[newaxis,...] * np.moveaxis(kernelCoords,d,0)
		self.fieldKernel[gridN, gridN] = self.fieldKernel[gridN + 1, gridN]
		self.fieldGradientKernel[:, gridN, gridN] = 0
		repStrength = 100*self.fieldKernel[gridN, gridN]
		repDist = 0.3*tumorModel.neighborDist
		self.fieldKernel -= repStrength*np.exp(-(kernelRs/repDist)**2)
		self.fieldGradientKernel -= (2*repStrength*np.exp(-(kernelRs/repDist)**2)/repDist**2)[newaxis,...] * np.moveaxis(kernelCoords,d,0)

		maxR = np.max(tumorModel.cellEffRadVec)
		self.displacementKernels = kernelCoords*(((1 + (1 - np.exp(-(kernelRs/maxR)**d))*(maxR / kernelRs)**d )**(1/d) - 1)/(maxR**d*sphereVol(d)))[..., newaxis]
		# self.displacementKernels[gridN, gridN, :] = 0
		#TODO finite diff check
		
		tumorCells = 1
		dt = 0.2 / max(np.max(list(tumorModel.growth.values())), np.max(tumorModel.diffusionMat), tumorModel.moveSpeed) #ensure the most common event only happens every 5 steps for small dt convergence
		
		if saveEvolution:
			self.evolution = []
		while tumorCells < tumorCellCount and (not saveEvolution or len(self.evolution)!=maxSteps):
			if saveEvolution:
				# if self.cellPositions.shape[1]>2:
					# ps = ps[np.all(np.abs(ps[:,2:])*2 < width, axis=1)] #just plot cells within slice
				inSlice = np.all(np.abs(self.cellPositions[:,2:]-L/2)*2 < width, axis=1)
				self.evolution.append((self.cellPositions[inSlice][:, :2], self.cellTypes[inSlice]))
			if verbose:
				print(f"{tumorCells} tumor cells: {100*np.log(tumorCells)/np.log(tumorCellCount):.1f}%")
				print(f"{np.sum(self.cellTypes == 2)} immune cells")
			tumorCells += self.update(tumorModel, dt, verbose)

	def addSources(self, sourcePoints, sourceAreas):
		diffs = self.cellPositions[:, newaxis,:] - sourcePoints[newaxis, :,:]
		ns = np.sum(diffs**2, axis=2)
		ns[ns==0] = 1
		sqrtArg = sourceAreas[newaxis,:]*(1-np.exp(-ns*np.pi/np.abs(sourceAreas)[newaxis,:]))/(ns*np.pi) + 1
		self.cellPositions += np.sum(diffs * (np.sqrt(sqrtArg) - 1)[:,:,newaxis], axis=1)

	def update(self, tumorModel, dt, profile=False, nThreads=-1):

		if profile: start = time.time()
		tree = cKDTree(self.cellPositions, compact_nodes=False)
		if profile: print(f"Make tree: {time.time() - start} s")
		if profile: start = time.time()
		pairs = tree.query_pairs(tumorModel.neighborDist, output_type="ndarray")
		if profile: print(f"Find pairs: {time.time() - start} s")

		# start = time.time()
		# NearestNeighbors(n_neighbors=20, radius=tumorModel.neighborDist, n_jobs=-1).fit(self.cellPositions).kneighbors(self.cellPositions, 2, return_distance=False)
		# print(f"Make tree sclearn: {time.time() - start} s")


		cancerCounts = np.histogramdd(self.cellPositions[self.cellTypes==1], bins = self.d*[self.gridEdges])[0]
		self.infection = fftconvolve(cancerCounts, self.fieldKernel, mode='same')
		with fft.set_workers(nThreads):
			self.infectionGradient = np.moveaxis([fftconvolve(cancerCounts, kernel, mode='same') for kernel in self.fieldGradientKernel],0,2) #scipy convolve cannot handle higher dim kernels..

		if profile: start = time.time()
		for i in range(len(types)):
			for j in range(len(types)):
				if tumorModel.diffusionMat[i,j]>0:
					allPairs = pairs[(self.cellTypes[pairs[:,0]]==i) & (self.cellTypes[pairs[:,1]]==j), :]
					n = np.random.binomial(allPairs.shape[0], dt * tumorModel.diffusionMat[i,j])
					choice = np.random.choice(allPairs.shape[0], size=n, replace=False)
					# for p in np.random.permutation(allPairs[np.random.rand(allPairs.shape[0]) < dt * tumorModel.diffusionMat[i,j]]): #permute to avoid finite dt effects being biased based on nn search order
					for p in allPairs[choice]: #permute to avoid finite dt effects being biased based on nn search order
						if self.cellTypes[p[0]]==i and self.cellTypes[p[1]]==j: #don't swap cells that have already been swapped..
							self.cellTypes[p[0]], self.cellTypes[p[1]] = self.cellTypes[p[1]], self.cellTypes[p[0]]
		if profile: print(f"Diffuse: {time.time() - start} s")

		onBorder = np.any(np.abs(self.cellPositions - self.L/2) > self.L/2 - tumorModel.neighborDist/2, axis=1)
		self.cellTypes[onBorder] = np.random.choice(3, size=np.sum(onBorder), replace=True, p=[1-tumorModel.immuneFraction, 0, tumorModel.immuneFraction])

		rad = 10
		diff = self.cellPositions[pairs[:,1]]-self.cellPositions[pairs[:,0]]
		diff2 = np.sum(diff*diff, axis=1)
		w = 5*np.exp(-diff2/rad**2)/rad**2
		f = 10*diff*w[:, newaxis]/np.sqrt(diff2)[:, newaxis]
		np.add.at(self.cellPositions, pairs[:,0], -f)
		np.add.at(self.cellPositions, pairs[:,1], f)

		# inside = np.all((self.cellPositions > 0) & (self.cellPositions < self.L), axis=1)
		# self.cellPositions, self.cellTypes = self.cellPositions[inside], self.cellTypes[inside]

		if profile: start = time.time()
		ccs = np.arange(len(self.cellPositions))[self.cellTypes==1]
		ics = np.concatenate( (pairs[(self.cellTypes[pairs[:,0]]==2) & (self.cellTypes[pairs[:,1]]==1), 0],
							   pairs[(self.cellTypes[pairs[:,1]]==2) & (self.cellTypes[pairs[:,0]]==1), 1]) )

		toSplit = np.concatenate( (ccs[np.random.rand(ccs.shape[0]) < dt*tumorModel.growth['cancer']], ics[np.random.rand(ics.shape[0]) < dt*tumorModel.growth['immune']] ))
		if profile: print(f"Finding to split: {time.time() - start} s")

		# self.addSources(self.cellPositions[toSplit,:], tumorModel.cellEffRadVec[self.cellTypes[toSplit]]**2*np.pi )

		if profile: start = time.time()
		expansionSources = np.histogramdd(self.cellPositions[toSplit], bins = self.d*[self.gridEdges], weights = sphereVol(self.d)*tumorModel.cellEffRadVec[self.cellTypes[toSplit]]**self.d, normed = False)[0]
		if profile: print(f"Calculating expansionSources: {time.time() - start} s")
		if profile: start = time.time()
		with fft.set_workers(nThreads):
			displacements = fftconvolve(np.broadcast_to(expansionSources[..., newaxis], self.d*[len(self.gridCenters)]+[self.d]), self.displacementKernels, mode='same', axes=range(self.d))
			# print(displacements[:4,:4,:])
		if profile: print(f"Calculating displacements: {time.time() - start} s")

		if True:
			toMove = np.arange(len(self.cellPositions))[self.cellTypes==2]
			toMove = toMove[np.random.rand(toMove.shape[0]) < dt*tumorModel.moveSpeed]
			# oldPositions = np.array(self.cellPositions[toMove])
			expansionSources -= np.histogramdd(self.cellPositions[toMove], bins = self.d*[self.gridEdges], weights = sphereVol(self.d)*tumorModel.cellEffRadVec[self.cellTypes[toMove]]**self.d, normed = False)[0]
			grads = interpn(self.d*[self.gridCenters], self.infectionGradient, self.cellPositions[toMove, :], bounds_error = False, fill_value=None)
			gradNorm = np.linalg.norm(grads, axis=1)[:, newaxis]
			gradNorm[gradNorm==0] = 1
			moveVecs = 8*grads / gradNorm
			self.cellPositions[toMove] += moveVecs
			
			expansionSources += np.histogramdd(self.cellPositions[toMove], bins = self.d*[self.gridEdges], weights = sphereVol(self.d)*tumorModel.cellEffRadVec[self.cellTypes[toMove]]**self.d, normed = False)[0]

			# self.addSources(np.concatenate([oldPositions, self.cellPositions[toMove,:]]), np.concatenate([-tumorModel.cellEffRadVec[self.cellTypes[toMove]]**2*np.pi, tumorModel.cellEffRadVec[self.cellTypes[toMove]]**2*np.pi]) )
		# self.addSources(self.cellPositions[toMove,:], tumorModel.cellEffRadVec[self.cellTypes[toMove]]**2*np.pi )
		
		if profile: start = time.time()
		self.cellPositions += interpn(self.d*[self.gridCenters], displacements, self.cellPositions, bounds_error = False, fill_value=None )
		if profile: print(f"Moving cells: {time.time() - start} s")
		
		# diffs = self.cellPositions[:, newaxis,:] - self.cellPositions[newaxis, toSplit,:]
		# ns = np.sum(diffs**2, axis=2)
		# ns[ns==0] = 1
		# r2s = tumorModel.cellEffRadVec[self.cellTypes[toSplit]]**2
		# self.cellPositions += np.sum(diffs * (np.sqrt(r2s[newaxis,:]/ns + 1) - 1)[:,:,newaxis], axis=1)

		if profile: start = time.time()
		randDirs = np.random.normal(size=(len(toSplit),self.d)).astype(np.float32)
		randDirs = tumorModel.cellEffRadVec[self.cellTypes[toSplit]][:, newaxis]*randDirs/np.linalg.norm(randDirs, axis=1)[:,newaxis]
		self.cellPositions = np.concatenate((self.cellPositions, self.cellPositions[toSplit] + randDirs), axis=0)
		self.cellPositions[toSplit] -= randDirs
		self.cellTypes = np.concatenate((self.cellTypes, self.cellTypes[toSplit]), axis=0)
		if profile: print(f"Adding cells: {time.time() - start} s")

		if profile: start = time.time()
		inside = np.all((self.cellPositions > 0) & (self.cellPositions < self.L), axis=1)
		self.cellPositions, self.cellTypes = self.cellPositions[inside], self.cellTypes[inside]
		if profile: print(f"Removing cells: {time.time() - start} s")

		return toSplit.shape[0]

def plot(cellPositions, cellTypes, width=10):
	colors = {'healthy':(0,.8,0), 'cancer':(1,0,0), 'immune':(0,0,1)}
	fig = pl.figure(figsize=(26,26))
	sc=[]
	for i in range(len(types)):
		ps = cellPositions[cellTypes==i]
		sc.append(pl.scatter(ps[:,0], ps[:,1], s=1, color=colors[types[i]], label=types[i]))
	pl.xlabel("x [μm]")
	pl.ylabel("y [μm]")
	pl.axis('square')
	pl.legend(loc="upper right")
	return fig, sc
	
if __name__ == "__main__":
	tm = TumorModel(immuneFraction=0.03, growth={'cancer':0.3, 'immune':0.1}, diffusion = {('cancer', 'immune'):0.0, ('cancer', 'healthy'):2.4, ('healthy','immune'):1.5})
	tm.moveSpeed = 4.0
	tumor = Tumor(tm, d=3, verbose=True, saveEvolution=True, L=5000, tumorCellCount=5_000_000, maxSteps=-1)
	
	fig, sc = plot(tumor.cellPositions, tumor.cellTypes)
	frameSkip = 1
	def updateFig(frameI):
		evI = frameI*frameSkip
		for i in range(len(types)):
			sc[i].set_offsets(tumor.evolution[evI][0][tumor.evolution[evI][1]==i])
		return sc
	import matplotlib.animation as animation
	anim = animation.FuncAnimation(fig, updateFig, frames=len(tumor.evolution)//frameSkip, blit=True)
	writervideo = animation.FFMpegWriter(fps=60) 
	anim.save(f"out/test.avi", writer=writervideo)
	# pl.show()
	#anim.save(f"out/test.gif", writer='imagemagick', fps=60)


	# pl.figure()
	# pl.imshow(tumor.infection[:,:])
	# pl.figure()
	# pl.imshow(tumor.infectionGradient[:,:,0])
	# pl.figure()
	# pl.imshow(tumor.infectionGradient[:,:,1])
	# pl.show()
