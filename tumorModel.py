import numpy as np
from numpy import newaxis
import matplotlib.pyplot as pl
from scipy.spatial import cKDTree
from scipy.special import gamma, k0, k1
from scipy.signal import fftconvolve
from scipy.interpolate import interpn
from scipy import fft, fftpack
from sklearn.neighbors import KDTree
from sklearn.neighbors import NearestNeighbors

import time
types = ['healthy', 'cancer', 'immune']


fields = ['']

def sphereVol(d):
	return np.pi**(d/2)/gamma(d/2 + 1)

class TumorModel():
	def __init__(self, cellEffR={t:10 for t in types}, immuneFraction=0, moveSpeed=0, cytotoxicity=0.0, neighborDist=25, growth={'cancer':1.,'immune':0.}, diffusion={}, infectionRange=200.):
		self.cellEffR = cellEffR
		self.cellEffRadVec = np.array([cellEffR[t] for t in types]).astype(np.float32)
		self.immuneFraction = immuneFraction
		self.growth = growth
		self.neighborDist = neighborDist
		self.diffusionMat = np.zeros([len(types)]*2)
		self.moveSpeed = moveSpeed
		self.cytotoxicity = cytotoxicity
		for p in diffusion:
			self.diffusionMat[types.index(p[0]), types.index(p[1])] = diffusion[p]
			self.diffusionMat[types.index(p[1]), types.index(p[0])] = diffusion[p]
		self.infectionRange = infectionRange

class Tumor():

	#fft of the last d axes
	def FFT(self, f):
		return fft.rfftn(f, s = self.d*(self.fftSize,))

	#last indices of arguments correspond to spatial grid postions.
	def convolve(self, aFFT, bFFT):
		return fft.irfftn( aFFT*bFFT, s = self.d*(self.fftSize,) )[(np.s_[...],) + self.d*(np.s_[self.gridN-1:2*self.gridN-1],)]

	def __init__(self, tumorModel, d=2, L=1500, tumorCellCount=1500, verbose=False, saveEvolution=False, maxSteps=-1, width=1):
		self.dtype = np.float32
		
		self.L = L
		self.d = d
		effVol = (sphereVol(d)*tumorModel.cellEffRadVec**d).dot( [1-tumorModel.immuneFraction, 0, tumorModel.immuneFraction] ) #fraction weighted average volume per cell
		nCells = int(L**d / effVol)
		nImmuneCells = int(nCells*tumorModel.immuneFraction)
		np.random.seed(0)
		self.cellPositions = (np.random.rand(nCells, d)*L).astype(self.dtype)
		self.cellTypes = np.zeros((nCells,), dtype=np.int32)
		

		self.cellTypes[:nImmuneCells] = 2
		self.cellPositions[nImmuneCells] = L/2
		self.cellTypes[nImmuneCells] = 1

		res = 12
		self.gridN = int(L/res + 0.5)
		self.fftSize = fftpack.next_fast_len(2*self.gridN - 1)
		gridDx = L/self.gridN
		kernelXs = np.linspace(-(self.gridN-1)*gridDx, (self.gridN-1)*gridDx, 2*self.gridN-1)
		self.gridEdges = np.linspace(0, L, self.gridN+1)
		self.gridCenters = (self.gridEdges[:-1] + self.gridEdges[1:])/2
		
		kernelCoords = np.moveaxis(np.meshgrid(*d*[kernelXs], indexing='ij'), 0, d)
		kernelRs = np.linalg.norm(kernelCoords, axis=d)
		kernelRsReg = np.array(kernelRs)
		middle = d*[self.gridN-1]
		kernelRsReg[tuple(middle)] = 1 #need to treat center specially
		
		self.infectionKernel = k0(kernelRsReg/tumorModel.infectionRange).astype(self.dtype)
		self.infectionGradientKernel = (-(k1(kernelRsReg/tumorModel.infectionRange)/(kernelRsReg*tumorModel.infectionRange))[newaxis,...] * np.moveaxis(kernelCoords,d,0)).astype(self.dtype)

		self.infectionKernel[tuple(middle)] = self.infectionKernel[tuple([self.gridN]+middle[1:])].astype(self.dtype)
		self.infectionGradientKernel[:, tuple(middle)] = 0

		
		repDist = 0.25*tumorModel.neighborDist
		self.exclusionKernel = np.exp(-(kernelRs/repDist)**2).astype(self.dtype)
		self.exclusionGradientKernel = ((2*np.exp(-(kernelRs/repDist)**2)/repDist**2)[newaxis,...] * np.moveaxis(kernelCoords,d,0)).astype(self.dtype)

		maxR = np.max(tumorModel.cellEffRadVec)
		self.displacementKernel = (kernelCoords*(((1 + (1 - np.exp(-(kernelRs/maxR)**d))*(maxR / kernelRsReg)**d )**(1/d) - 1)/(maxR**d*sphereVol(d)))[..., newaxis]).astype(self.dtype)

		#TODO finite diff check

		with fft.set_workers(-1):
			#precompute FFTs of kernels for faster convolutions
			if tumorModel.moveSpeed:
				self.infectionKernelFFT = self.FFT(self.infectionKernel)
				self.infectionGradientKernelFFT = self.FFT(self.infectionGradientKernel)
				self.exclusionKernelFFT = self.FFT(self.exclusionKernel)
				self.exclusionGradientKernellFFT = self.FFT(np.moveaxis(self.exclusionGradientKernel,d,0))
			self.displacementKernelFFT = self.FFT(np.moveaxis(self.displacementKernel,d,0))
		
		tumorCells = 1
		dt = 0.2 / max(np.max(list(tumorModel.growth.values())), np.max(tumorModel.diffusionMat), tumorModel.moveSpeed) #ensure the most common event only happens every 5 steps for small dt convergence
		
		if saveEvolution:
			self.evolution = []
		i = 0 
		while tumorCells < tumorCellCount and i != maxSteps:
			if saveEvolution:
				# if self.cellPositions.shape[1]>2:
					# ps = ps[np.all(np.abs(ps[:,2:])*2 < width, axis=1)] #just plot cells within slice
				inSlice = np.all(np.abs(self.cellPositions[:,2:]-L/2)*2 < width, axis=1)
				self.evolution.append((self.cellPositions[inSlice][:, :2], self.cellTypes[inSlice]))
			if verbose:
				print(f"{tumorCells} tumor cells: {100*np.log(tumorCells)/np.log(tumorCellCount):.1f}%")
				print(f"{np.sum(self.cellTypes == 2)} immune cells")
			tumorCells += self.update(tumorModel, dt, verbose)
			i+=1

# 33699578 tumor cells: 99.9%
# 7879667 immune cells
# Make tree: 38.672526597976685 s
# Find pairs: 116.22024607658386 s
# Counting cancer cells: 5.646864891052246 s
# Diffuse: 121.81452488899231 s
# Set boundary condition: 0.6678287982940674 s
# Measure neighbor distances: 97.12076663970947 s
# Calculate neighbor displacements: 35.956838607788086 s
# Add neighbor displacements: 63.323172092437744 s
# 7848221 cancer cells cannot split
# Finding to split: 80.93684267997742 s
# Calculating expansionSources: 0.3249225616455078 s
# Calculating displacements: 2.28892183303833 s
# Displacing cells: 15.540356397628784 s
# Adding cells: 0.3608520030975342 s
# Removing cells: 1.3589246273040771 s


	def update(self, tumorModel, dt, profile=False, nThreads=-1):

		if profile: start = time.time()
		tree = cKDTree(self.cellPositions, compact_nodes=False)
		if profile: print(f"Make tree: {time.time() - start} s")
		if profile: start = time.time()
		pairs = tree.query_pairs(tumorModel.neighborDist, output_type="ndarray")
		if profile: print(f"Find pairs: {time.time() - start} s")

		if profile: start = time.time()
		cancerCounts = np.histogramdd(self.cellPositions[self.cellTypes==1], bins = self.d*[self.gridEdges])[0].astype(self.dtype)
		if profile: print(f"Counting cancer cells: {time.time() - start} s")

		if tumorModel.moveSpeed>0:
			if profile: start = time.time()
			with fft.set_workers(nThreads):
				cancerCountsFFT = self.FFT(cancerCounts)
				self.infection = self.convolve(cancerCountsFFT, self.infectionKernelFFT)
				self.infectionGradient = self.convolve(cancerCountsFFT, self.infectionGradientKernelFFT)
				self.exclusion = self.convolve(cancerCountsFFT,  self.exclusionKernelFFT)
			if profile: print(f"Calculating infection and exclusion fields: {time.time() - start} s")

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

		if profile: start = time.time()
		onBorder = np.any(np.abs(self.cellPositions - self.L/2) > self.L/2 - tumorModel.neighborDist/2, axis=1)
		self.cellTypes[onBorder] = np.random.choice(3, size=np.sum(onBorder), replace=True, p=[1-tumorModel.immuneFraction, 0, tumorModel.immuneFraction])
		if profile: print(f"Set boundary condition: {time.time() - start} s")

		if profile: start = time.time()
		
		diff = self.cellPositions[pairs[:,1]]-self.cellPositions[pairs[:,0]]
		radSum = tumorModel.cellEffRadVec[self.cellTypes[pairs[:,0]]] + tumorModel.cellEffRadVec[self.cellTypes[pairs[:,1]]]
		diff2 = np.sum(diff*diff, axis=1)
		if profile: print(f"Measure neighbor distances: {time.time() - start} s")
		
		rad = 10
		if profile: start = time.time()
		# w = 10*np.exp(-diff2/rad**2)/rad**(self.d/2)
		# f = diff*w[:, newaxis]/np.sqrt(diff2)[:, newaxis] #10
		eps = 1e-6
		maxStepSize = 0.5 #2 um per step. This is set by the overall scale, too small is slow, too large might be unstable. No dt
		f = maxStepSize * diff * np.exp(-diff2/radSum**2)[:, newaxis] / np.sqrt(eps + diff2)[:, newaxis]
		if profile: print(f"Calculate neighbor displacements: {time.time() - start} s")

		if profile: start = time.time()
		# np.add.at(self.cellPositions, pairs[:,0], -f) #these are crazy slow
		# np.add.at(self.cellPositions, pairs[:,1], f) #so we use the workaround below
		for i in range(self.d):
			self.cellPositions[:,i] -= np.bincount(pairs[:,0], weights=f[:,i], minlength=self.cellPositions.shape[0])
			self.cellPositions[:,i] += np.bincount(pairs[:,1], weights=f[:,i], minlength=self.cellPositions.shape[0])
		if profile: print(f"Add neighbor displacements: {time.time() - start} s")

		if profile: start = time.time()
		cancerCells = np.arange(len(self.cellPositions))[(self.cellTypes==1)]
		immuneCellsNearCancer = np.concatenate( (pairs[(self.cellTypes[pairs[:,0]]==2) & (self.cellTypes[pairs[:,1]]==1), 0],
							   pairs[(self.cellTypes[pairs[:,1]]==2) & (self.cellTypes[pairs[:,0]]==1), 1]) )
		cancerCellsNearImmune = np.concatenate( (pairs[(self.cellTypes[pairs[:,0]]==2) & (self.cellTypes[pairs[:,1]]==1), 1],
							   pairs[(self.cellTypes[pairs[:,1]]==2) & (self.cellTypes[pairs[:,0]]==1), 0]) )
		
		if tumorModel.cytotoxicity>0:
			splittableCancerCells = np.setdiff1d(cancerCells, cancerCellsNearImmune[np.random.rand(cancerCellsNearImmune.shape[0]) < tumorModel.cytotoxicity])
			print(f'{len(cancerCells) - len(splittableCancerCells)} cancer cells cannot split')
		else:
			splittableCancerCells = cancerCells

		toSplit = np.concatenate(  (splittableCancerCells[np.random.rand(splittableCancerCells.shape[0]) < dt*tumorModel.growth['cancer']],
									immuneCellsNearCancer[np.random.rand(immuneCellsNearCancer.shape[0]) < dt*tumorModel.growth['immune']] ))
		if profile: print(f"Finding to split: {time.time() - start} s")

		if profile: start = time.time()
		expansionSources = np.histogramdd(self.cellPositions[toSplit], bins = self.d*[self.gridEdges], weights = sphereVol(self.d)*tumorModel.cellEffRadVec[self.cellTypes[toSplit]]**self.d,
			normed = False)[0].astype(self.dtype)
		if profile: print(f"Calculating expansionSources: {time.time() - start} s")
		
		if tumorModel.moveSpeed>0:
			if profile: start = time.time()
			toMove = np.arange(len(self.cellPositions))[self.cellTypes==2]
			toMove = toMove[np.random.rand(toMove.shape[0]) < dt*tumorModel.moveSpeed]
			expansionSources -= np.histogramdd(self.cellPositions[toMove], bins = self.d*[self.gridEdges], weights = sphereVol(self.d)*tumorModel.cellEffRadVec[self.cellTypes[toMove]]**self.d, normed = False)[0].astype(self.dtype)
			grads = interpn(self.d*[self.gridCenters], np.moveaxis(self.infectionGradient, 0, self.d), self.cellPositions[toMove, :], bounds_error = False, fill_value=None)
			gradNorm = np.linalg.norm(grads, axis=1)[:, newaxis]
			gradNorm[gradNorm==0] = 1
			moveVecs = 5*grads / gradNorm
			exclusionThreshold = 0.5
			exclusionLevels = interpn(self.d*[self.gridCenters], self.exclusion, self.cellPositions[toMove, :] + moveVecs, bounds_error = False, fill_value=None)
			self.cellPositions[toMove[exclusionLevels < exclusionThreshold]] += moveVecs[exclusionLevels < exclusionThreshold]
			expansionSources += np.histogramdd(self.cellPositions[toMove], bins = self.d*[self.gridEdges], weights = sphereVol(self.d)*tumorModel.cellEffRadVec[self.cellTypes[toMove]]**self.d, normed = False)[0].astype(self.dtype)
			if profile: print(f"Moving immune cells: {time.time() - start} s")
		else:
			toMove = []

		if len(toMove) + len(toSplit) > 0:
			if profile: start = time.time()
			with fft.set_workers(nThreads):
				expansionSourcesFFT = self.FFT(expansionSources)
				displacements = self.convolve(expansionSourcesFFT, self.displacementKernelFFT)
			if profile: print(f"Calculating displacements: {time.time() - start} s")

			if profile: start = time.time()
			self.cellPositions += interpn(self.d*[self.gridCenters], np.moveaxis(displacements, 0, self.d), self.cellPositions, bounds_error = False, fill_value=None, method='nearest' )
			if profile: print(f"Displacing cells: {time.time() - start} s")
		
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

def plot(cellPositions, cellTypes, cellEffR, L=0, width=15, fig=None):
	colors = {'healthy':(0,.8,0), 'cancer':(1,0,0), 'immune':(0,0,1)}
	if fig==None:
		fig = pl.figure(figsize=(26,26))
	sc=[]
	import matplotlib
	for i in range(len(types)):
		ps = cellPositions[cellTypes==i]
		inSlice = np.all(np.abs(ps[:,2:]-L/2)*2 < width, axis=1)
		# sc.append(pl.scatter(ps[:,0], ps[:,1], s=1, color=colors[types[i]], label=types[i]))
		circles = [pl.Circle((xi,yi), radius=cellEffR[types[i]]*0.3, linewidth=0, label=types[i]) for xi,yi in ps[inSlice][:, :2]]
		# print(circles)
		c = matplotlib.collections.PatchCollection(circles, color=colors[types[i]] )
		sc.append(c)
		pl.gca().add_collection(c)
	pl.xlabel("x [μm]")
	pl.ylabel("y [μm]")
	pl.axis('square')
	pl.legend(loc="upper right")
	return fig, sc
	
if __name__ == "__main__":
	tm = TumorModel(immuneFraction=0.03, growth={'cancer':1.0, 'immune':0.1}, diffusion = {('cancer', 'immune'):0.0, ('cancer', 'healthy'):0.2, ('healthy','immune'):0.3})
	tm.moveSpeed = 1.0
	tumor = Tumor(tm, d=3, verbose=True, saveEvolution=True, L = 3000., tumorCellCount = 20_000_000, maxSteps=-1, width=10)
	# tumor = Tumor(tm, d=3, verbose=True, saveEvolution=True, L = 1000., tumorCellCount=100_000, maxSteps=3000, width=10)
	# tumor = Tumor(tm, d=2, verbose=True, saveEvolution=True, L = 2200., tumorCellCount=100_000, maxSteps=3000, width=10)
	fig, sc = plot(tumor.cellPositions, tumor.cellTypes, tm.cellEffR)
	pl.savefig("out/test.png")

	frameSkip = 1
	def updateFig(frameI):
		evI = frameI*frameSkip
		for i in range(len(types)):
			circles = [pl.Circle((xi,yi), radius=tm.cellEffR[types[i]]*0.3, linewidth=0, label=types[i]) for xi,yi in tumor.evolution[evI][0][tumor.evolution[evI][1]==i]]
			sc[i].set_paths(circles)
			# sc[i].set_offsets(tumor.evolution[evI][0][tumor.evolution[evI][1]==i])
		return sc
	import matplotlib.animation as animation
	anim = animation.FuncAnimation(fig, updateFig, frames=len(tumor.evolution)//frameSkip, blit=True)
	writervideo = animation.FFMpegWriter(fps=30) 
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
