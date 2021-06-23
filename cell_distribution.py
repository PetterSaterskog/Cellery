import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix, lil_matrix
from scipy.fftpack import dst, dct, next_fast_len
from multiprocessing import Pool
import math
from collections import defaultdict

# This gives the probability that two points placed in the unit square with
# uniform distributions has a squared distance smaller than s2.
# See e.g. https://people.kth.se/~johanph/habc.pdf on how to generalize this to non-square images.
def gCum(r):
	# return -4 * np.sqrt(s2) + np.pi + s2 if s2 < 1 else -2 + 4 * np.arcsin(1/np.sqrt(s2)) + 4*np.sqrt(s2-1) - np.pi - s2
	return np.pi*r**2-8*r**3/3+r**4/2 if r<1 else 1/3 - 2 * r**2 - np.pi*r**2 - r**4/2 + 4*np.sqrt(r**2 - 1) + 8/3*(r**2-1)**(3/2) + 4*r**2*np.arcsin(1/r)

def g(r):
	r2 = r*r
	return 2*r*(-4 * r + np.pi + r2 if r2 <= 1 else -2 + 4 * np.arcsin(1/r) + 4*np.sqrt(r2-1) - np.pi - r2)

def cellDiff(x1, x2, W, torus):
	d = x2 - x1
	if torus:
		d = np.remainder(d + W/2, W) - W/2
	return d

def getPairGrad(arg):
	pair, cells, trees, qs, W, z, torus, gradient = arg
	maxDist = 2*np.pi/qs[0]
	c1, c2 = [cells[t] for t in pair]
	# if pair[0]==pair[1]:
	# 	cellPairs = trees[pair[0]].query_pairs(maxDist, output_type = 'ndarray')
	# else:
	# 	neighbors = trees[pair[0]].query_ball_tree(trees[pair[1]], maxDist)
	# 	cellPairs = np.array([(i, j) for i in range(len(c1)) for j in neighbors[i]])
	
	if False and len(cellPairs) == 0:
		diff = np.zeros((0,2))
		if gradient:
			return qs*0, [ [ np.zeros((len(cells[pair[ci]]), len(qs)))  for di in range(2)] for ci in range(2)]
		else:
			return qs*0,
	else:
		# diff = cellDiff(c1[cellPairs[:,0]], c2[cellPairs[:,1]], W, torus)
		diff = cellDiff(c1[:, np.newaxis], c2[np.newaxis, :], W, torus).reshape(-1, 2)
	dists = np.linalg.norm(diff, axis=1)
	dists = dists[dists>1] #remove self-pairs

	if dists.shape[0] == 0:
		diff = np.zeros((0,2))
		if gradient:
			return qs*0, [ [ np.zeros((len(cells[pair[ci]]), len(qs)))  for di in range(2)] for ci in range(2)]
		else:
			return qs*0,

	

		# ddistdx[ti][di][cell, dist]
	# l2 =  4/qs[0]**2
	# dqrdx = [ [ddistdx[ci][di][:, np.newaxis] * qs[np.newaxis, :] for di in range(2)] for ci in range(2)]
	dq = qs[1]-qs[0]

	n = len(qs)

	batchSize = 10000 #batch this sum to not run out fo memory
	S = 0
	if True:
		edges = np.linspace(0, W*np.sqrt(2), 10*n)
		binCenters = (edges[:-1]+edges[1:])/2
		counts,_ = np.histogram(dists, edges)
		weights = np.array([(2*np.pi*(d/W))/g(d/W) for d in binCenters])
		normCounts = counts*weights
		uniform = np.diff(np.pi*edges**2)*c1.shape[0]*c2.shape[0]/W**2
		interp = 1/(1+np.exp(-(binCenters-0.5*W)/(0.1*W)))
		normCounts = normCounts*(1-interp) + interp*uniform
		normCounts[binCenters>W*1.2] = uniform[binCenters>W*1.2]
		# import matplotlib.pyplot as pl
		# pl.figure()
		# pl.plot(binCenters,counts)
		rRes = self.W/np.sqrt(len(self.cells[pair[0]])*len(self.cells[pair[1]]))


		S += np.sum( normCounts[:, np.newaxis]*np.sin( binCenters[:, np.newaxis] * qs[np.newaxis, :] )*np.exp(-binCenters[:, np.newaxis]**2/(2/dq**2)), axis=0)
	else:
		if torus:
			weights = np.ones(dists.shape)
		else:
			assert(gradient==False) #not handling derivative of this..
			weights = np.array([(2*np.pi*(d/W))/g(d/W) for d in dists])
			weights = np.diff(np.pi*edges**2)/np.diff(gCum(edges/W)*W*W)
		for i in range(1+dists.shape[0]//batchSize):
			S += np.sum( weights[i*batchSize:(i+1)*batchSize, np.newaxis]*np.sin( dists[i*batchSize:(i+1)*batchSize, np.newaxis] * qs[np.newaxis, :] )*np.exp(-dists[i*batchSize:(i+1)*batchSize, np.newaxis]**2/(2/dq**2)), axis=0)
	S *=  2 / qs / z
	# S = np.sum( weights[:, np.newaxis]*np.sin( dists[:, np.newaxis] * qs[np.newaxis, :] )*np.exp(-dists[:, np.newaxis]**2/(2*l2)) / qs[np.newaxis, :] / dists[:, np.newaxis], axis=0)
	if gradient:
		ddistdx = [ [csr_matrix(([-1,1][ci]*diff[:,di]/dists, (cellPairs[:, ci], np.arange(len(dists)) )), shape=(len(cells[pair[ci]]), len(dists))) for di in range(2)] for ci in range(2)]
		return S, [ [ ddistdx[ci][di].dot(np.cos( dists[:, np.newaxis] * qs[np.newaxis, :] )*np.exp(-dists[:, np.newaxis]**2/(2*l2)) / dists[:, np.newaxis]
                    -  np.sin( dists[:, np.newaxis] * qs[np.newaxis, :] )*np.exp(-dists[:, np.newaxis]**2/(2*l2)) / qs[np.newaxis, :]/dists[:, np.newaxis]**2
		    - np.sin( dists[:, np.newaxis] * qs[np.newaxis, :] )*np.exp(-dists[:, np.newaxis]**2/(2*l2)) / qs[np.newaxis, :] /l2)  for di in range(2)] for ci in range(2)]
	else:
		return S,

# We define fourier transfom as:
#	F(q) = int d^3r f(r) exp(-iq dot r)
#	f(r) = int d^3q F(q) exp(iq dot r) / (2pi)^3
# The functions are rotationally symmetric so the angular part can be done. The remaining radial part becomes a sine transform
# that is implemented through a discreticed integral in the two functions below. They are exact inverses of each other

# sum i f(i dqr ) * dq * q * sin(dr * dq * i * j + ..) / (r 2 pi^2)
def radialFourierTransformQToR(qs, fs, deWindow = False, rResolution=0):
	n = len(qs)
	dq = qs[1]-qs[0]
	rs = (0.5+np.arange(n))*np.pi/n/dq
	dr = rs[1]-rs[0]
	# return rs, dst(fs * qs * np.exp(-0*qs**2*rResolution**2), type=4)*(np.exp(rs*rs*dq**2/2) if deWindow else 1) * dq  / (4*np.pi**2)
	return rs, dst(fs * qs * np.exp(-0*2*qs**2/(qs[-1]**2)), type=4)*(np.exp(0*rs*rs*dq**2/2) if deWindow else 1) * dq / rs / (4*np.pi**2)

# sum i f(i dr ) * dr * r * sin(dr * dq * i * j + ..) 4 pi / q
def radialFourierTransformRToQ(rs, fs, deWindow = False):
	n = len(rs)
	dr = rs[1]-rs[0]
	qs = (0.5+np.arange(n))*np.pi/n/dr
	dq = qs[1]-qs[0]
	return qs, dst(fs * rs * np.exp(-0*rs*rs*dq**2/2), type=4) / qs * np.exp(0*2*qs**2/(qs[-1]**2)) * dr * (2*np.pi)
	#dst = 2*sin(..)
	# 4 pi / (2 pi)^3 = 1/(2 pi^2)
	# n = len(rs)
	# qs = np.arange(1,n+1)*np.pi/(n+1)/rs[0]
	# return qs, dst(fs * rs, type=1) / qs 


def getRadialConvolutionMatrix(dr, n, fs):
	inds = np.arange(n, dtype = np.int32)
	rs = (0.5 + inds) * dr
	indsSum = inds[:, np.newaxis] + inds[np.newaxis, :]
	indsDiff = np.abs(inds[:, np.newaxis] - inds[np.newaxis, :])
	frsCum = np.cumsum(np.concatenate(([0], fs*rs, np.zeros(n))))
	return 2 * np.pi * dr**2 * rs[np.newaxis, :] * (frsCum[indsSum + 1] - frsCum[indsDiff]) / rs[:, np.newaxis]

class CellDistribution():
	def __init__(self, cells, W, z = 1, torus = False):
		self.cells = {t:np.array(cells[t]) for t in cells}
		self.W, self.z = W, z
		self.torus = torus
		self.trees = {t:cKDTree(cells[t], boxsize = (W, W) if torus else None) for t in cells}
	
	def cellDiff(self, x1, x2):
		d = x2 - x1
		if self.torus:
			d = np.remainder(d + self.W/2, self.W) - self.W/2
		return d

	#This returns g for all cell type pairs, the results are regularized where data is missing. g(0)=0 and g(inf)=1 are assumed.
	#Perhaps input expected cell sizes to this for better guesses?
	def getG(self, dr, n):
		cellCounts = self.getCounts()
		rhos = self.getRhos()

		edges = np.arange(n+1)*dr
		rs = (0.5+np.arange(n))*dr

		gridResolution = 3 #assume this is fixed
		gridEdges = np.linspace(0, self.W, gridResolution+1)
		gridCellW = gridEdges[1] - gridEdges[0]
		assert(gridCellW*np.sqrt(2) >= edges[-1])
		ps = self.cells
		nComps = len(self.cells)*( len(self.cells) + 1 )*n//2
		M  = [np.zeros((gridResolution**2+1)//2, nComps), np.zeros((gridResolution**2-1)//2, nComps)]
		groupI = 2*[0]
		for i in range(gridResolution**2):
			group = (i//gridResolution + i%gridResolution) % 2
			(x0, x1), (y0, y1) = gridEdges[i//gridResolution:i//gridResolution+2], gridEdges[i%gridResolution:i%gridResolution+2]
			j=0
			for t1 in self.cells:
				for t2 in self.cells:
					if t1 <= t2:
						c1, c2 = [ps[(ps[:,0] >= x0) & (ps[:,0] < x1) & (ps[:,1] >= y0) & (ps[:,1] < y1)] for ps in [self.cells[t1], self.cells[t2]]]
						diff = self.cellDiff(c1[:, np.newaxis], c2[np.newaxis, :]).reshape(-1, 2)
						dists = np.linalg.norm(diff, axis=1)
						counts, _ = np.histogram(dists[dists>1], edges)
						M[group, j:j+n, i] = counts / (np.diff([gCum(e/gridCellW)*gridCellW*gridCellW for e in edges]) * len(c1) * len(c2) / gridCellW**2)
						j += n
			groupI[group] += 1
		cov = np.mean([np.cov(M[g, :, :]) for g in range(2)])
		np.linalg.eigh(cov)

		np.ei

		import matplotlib.pyplot as pl
		from matplotlib.colors import hsv_to_rgb
		pl.figure()

		for i in range(gridResolution):
			for j in range(gridResolution):
				for t1 in self.cells:
					for t2 in self.cells:
						if t1 <= t2:
							pl.plot(rs, gridGs[i][j][(t1, t2)], color = hsv_to_rgb(((i+gridResolution*j)/gridResolution**2,1,1)))
		
		pl.show()
		exit(0)

		#Distances longer than this are too influenced by boundary effects in crop case
		#and finite size effects in toroidal case. We set such values to 1
		maxMeasR = self.W*0.8
		transitionWidth = self.W*0.10
		nTransitionsCutoff = 3

		g = {}
		for t1 in self.cells:
			for t2 in self.cells:
				if t1 <= t2:
					c1 = self.cells[t1]
					c2 = self.cells[t2]
					diff = self.cellDiff(c1[:, np.newaxis], c2[np.newaxis, :]).reshape(-1, 2)
					dists = np.linalg.norm(diff, axis=1)
					dists = dists[dists>0]
					counts,_ = np.histogram(dists, edges)
					gMeas = counts / (np.diff([gCum(e/self.W)*self.W*self.W for e in edges]) * self.z * rhos[t1] * cellCounts[t2])
					interp = 1/(1+np.exp(-(rs-(maxMeasR - nTransitionsCutoff*transitionWidth))/transitionWidth))
					interp[rs>maxMeasR] = 1

					gInterp = gMeas*(1 - interp) + 1*interp
					
					rRes = 2*self.W/np.sqrt(cellCounts[t1]*cellCounts[t2])
					kernelN = math.ceil(4*rRes / dr)
					kernel = np.exp(-np.linspace(-kernelN*dr, kernelN*dr, 2*kernelN+1)**2 / (2*rRes**2))
					kernel /= np.sum(kernel)
					g[(t1,t2)] = gInterp
					# g[(t1,t2)] = np.convolve(np.concatenate([np.zeros(kernelN), gInterp, np.ones(kernelN)]), kernel, mode='valid')
		return g

	def getCounts(self):
		return {t:self.cells[t].shape[0] for t in self.cells}	
	
	def getRhos(self):
		return {t:self.cells[t].shape[0]/self.W**2/self.z for t in self.cells}

	def getCorrs(self, edges):
		if self.torus:
			assert( edges[-1] <= self.W/2 ) #longer than this and we need to consider multiple paths
			genAreas = np.pi*edges**2
		else:
			assert( edges[-1] <= self.W*np.sqrt(2) )
			genAreas = [gCum(e/self.W) for e in edges]
		# print(genAreas)
		uniformProbabilitiesPerBin = np.diff(genAreas) # / self.W**2
		corrs, errs = {}, {}
		for t1 in self.trees:
			for t2 in self.trees:
				if t1 > t2: continue
				c1, c2 = [self.cells[t] for t in [t1,t2]]
				if t1==t2:
					if len(c1) < 2: continue
					pairs = self.trees[t1].query_pairs(edges[-1], output_type = 'ndarray')
					if len(pairs):
						norm = 2 / uniformProbabilitiesPerBin / len(c1) / (len(c1)-1)
					else: norm = 1
				else:
					neighbors = self.trees[t1].query_ball_tree(self.trees[t2], edges[-1])
					pairs = np.array([(i, j) for i in range(len(c1)) for j in neighbors[i]])
					if len(pairs):
						norm = 1 / uniformProbabilitiesPerBin / len(c1) / len(c2)
					else: norm = 1
				if len(pairs) == 0:
					dists = []
				else:
					dists = cellDiff(c1[pairs[:,0]], c2[pairs[:,1]], self.W, self.torus)
				hist = np.histogram(dists, edges)[0]
				corrs[(t1,t2)] =  hist * norm
				errs[(t1,t2)] = np.sqrt(hist + 1) * norm
		return corrs, errs

	# This returns structure factor - 1, we do not count cells distances to themselves.
	def getStructureFactors(self, qs, gradient=False, nThreads = 1):
		print("Calculating structure factor...")
		rs = self.getRs(qs)
		_, gs = self.getG(rs[1]-rs[0], len(qs))
		

		# pairs = [(t1, t2) for t1 in self.trees for t2 in self.trees if t1<=t2 and not (t1==t2 and len(self.cells[t1])==0)]
		# if nThreads>1:
		# 	with Pool(nThreads) as pool:
		# 		res = list(zip(*pool.map(getPairGrad, [(p, self.cells, self.trees, qs, self.W, self.z, self.torus, gradient) for p in pairs])))
		# else:
		# 	res = list(zip(*[getPairGrad((p, self.cells, self.trees, qs, self.W, self.z, self.torus, gradient)) for p in pairs]))
		
		# if gradient:
		# 	S = dict(zip(pairs, res[0]))
		# 	dSdx = dict(zip(pairs, res[1]))
		# 	return S, dSdx
		# else:
		# 	return [dict(zip(pairs, res[0]))]
	
	def getQs(self, resolution):
		maxr = self.W * np.sqrt(2) #*2, <- works fine with larger values too, though its slower. Perhaps good for less aliasing?
		n = next_fast_len(int(2*maxr/resolution+1))
		return (0.5+np.arange(n)) * np.pi / maxr

	def getRs(self, qs):
		dq = qs[1]-qs[0]
		n = len(qs)
		return (0.5+np.arange(n))*np.pi/n/dq

	def getH(self, qs, S=None):
		rs = self.getRs(qs)
		_, gs = self.getG(rs[1]-rs[0], len(qs))
		return {p:radialFourierTransformRToQ(rs, gs[p]-1)[1] for p in gs}
		
		# if S==None:
		# 	S, = self.getStructureFactors(qs)
		# counts = self.getCounts()
		# rhos = self.getRhos()
		# Sn = {p:S[p]/(counts[p[0]]*rhos[p[1]]) for p in S}
		# rs = self.getRs(qs)
		# dq = qs[1] - qs[0]
		# window = np.exp(-rs**2/(2/dq**2))
		# window =  np.ones(len(rs))
		# _,deltaFun = radialFourierTransformRToQ(rs, window)

		# # import matplotlib.pyplot as pl
		# # pl.figure()
		# # for p in Sn:
		# # 	pl.plot(rs, Sn[p], label=p)
		# # pl.plot(rs, deltaFun, label="delta")
		# # pl.legend()

		# # pl.figure()
		# # for p in Sn:
		# # 	pl.plot(rs, radialFourierTransformQToR(qs, Sn[p])[1], label=p)
		# # pl.plot(rs, radialFourierTransformQToR(qs, deltaFun)[1], label="delta")
		# # pl.legend()
		# # pl.show()
		# # exit(0)

		# return {p:Sn[p]-deltaFun for p in S}

	def getRRes(self, pair):
		return self.W/np.sqrt(len(self.cells[pair[0]])*len(self.cells[pair[1]]))

	def getRadialDistributionFunction(self, resolution):
		qs = self.getQs(resolution)

		print("Preparing matrices...")
		types = list(self.cells.keys())
		rhos = self.getRhos()
		rhoVec = np.array([rhos[t] for t in types])
		
		# S, = self.getStructureFactors(qs)
		H = self.getH(qs)
		print(H)
		Hmat = np.array([[[H[(min(t1,t2), max(t1, t2))][i] for t2 in types] for t1 in types] for i in range(len(qs))])

		print("Solving...")
		# h = d + rho h d
		DMat = np.linalg.solve( np.eye(len(types))[np.newaxis,:,:] + rhoVec[np.newaxis, np.newaxis, :]*Hmat, Hmat)
		# DMat = np.linalg.solve( np.eye(len(types))[np.newaxis,:,:] , Smat)
		
		counts = self.getCounts()
		rhos = self.getRhos()
		# Sn = {p:S[p]/(counts[p[0]]*rhos[p[1]]) for p in S}
		h = {p:radialFourierTransformQToR(qs, H[p]) for p in H}
		C = {p:radialFourierTransformQToR(qs, DMat[:, types.index(p[0]), types.index(p[1])]) for p in H} #no dewindow, expect to go to 0 anyways..
		# return g, C, rhoVec[np.newaxis, np.newaxis, :]*Hmat,  DMat
		return h, C, rhoVec[np.newaxis, np.newaxis, :]*Hmat

	def __getBasisXInd(self, nt, nb):
		types = list(self.cells.keys())
		return [(t1,t2, bi) for t1 in range(nt) for t2 in range(nt) for bi in range(nb) if types[t1]<=types[t2]]

	def getCFromBasis(self, x, basis, dr, n):
		xInds =  self.__getBasisXInd(len(self.cells), len(basis))
		c = defaultdict(lambda: np.zeros((n)))
		rs = (0.5+np.arange(n)) * dr
		types = list(self.cells.keys())
		for i in range(len(xInds)):
			c[(types[xInds[i][0]], types[xInds[i][1]])] += x[i]*basis[xInds[i][2]]
		return c

	#
	def getC(self, dr, n, basis = None, longRangeReg = 0, highFreqReg = 0):
		gs = self.getG(dr, n)

		dq=np.pi/n/dr
		qs = (0.5+np.arange(n)) * dq
		rs = (0.5+np.arange(n)) * dr

		types = list(self.cells.keys())
		nt = len(types)
		rhos = self.getRhos()
		rhoVec = np.array([rhos[t] for t in types])
		
		if longRangeReg>0 or highFreqReg>0:
			h = np.zeros((n, len(types), len(types)))
			for i in range(len(types)):
				for j in range(len(types)):
					h[:,i,j] = gs[(min(types[i], types[j]), max(types[i], types[j]))] - 1

			xInd_To_CT1T2Q = [(t1, t2, qi) for t1 in range(nt) for t2 in range(nt) for qi in range(n) if types[t1]<=types[t2]]
			CT1T2Q_To_xInd = {**{xInd_To_CT1T2Q[i]:i for i in range(len(xInd_To_CT1T2Q))}, **{(xInd_To_CT1T2Q[i][1], xInd_To_CT1T2Q[i][0], xInd_To_CT1T2Q[i][2]):i for i in range(len(xInd_To_CT1T2Q))}}

			nComps = nt*(nt+1)//2
			nEqs = nt**2*n + nComps*(n-1) + nComps*n
			A = np.zeros(( nEqs, nComps*n))
			b = np.zeros(( nEqs,))
			eqInd = 0
			for t2Eq in range(nt):
				for t1Eq in range(nt):
					for t1C in range(nt):
						xInd = CT1T2Q_To_xInd[(t1C, t2Eq, 0)]
						A[eqInd:eqInd+n, xInd:xInd+n] += getRadialConvolutionMatrix(dr, n, rhoVec[t1C]*h[:, t1Eq, t1C])
						if t1Eq==t1C:
							A[eqInd:eqInd+n, xInd:xInd+n] += np.eye(n)
						b[eqInd:eqInd+n] = h[:, t1Eq, t2Eq]
					eqInd += n
			
			for t2C in range(nt):
				for t1C in range(t2C+1):
					xInd = CT1T2Q_To_xInd[(t1C, t2C, 0)]
					A[eqInd:eqInd+n-1, xInd:xInd+n] = 0*highFreqReg / np.sqrt(dr) * (np.diag(np.ones(n)) - np.diag(np.ones(n-1), 1))[:-1, :] * rs[1:][:, np.newaxis]/20
					eqInd += n-1
			
			for t2C in range(nt):
				for t1C in range(t2C+1):
					xInd = CT1T2Q_To_xInd[(t1C, t2C, 0)]
					A[eqInd:eqInd+n, xInd:xInd+n] = 0*longRangeReg * np.sqrt(dr) * np.eye(n) * rs**2
					eqInd += n

			assert(eqInd == A.shape[0])

			# import matplotlib.pyplot as pl
			# pl.imshow(A==0)
			# pl.show()
			# exit(0)
			print("Solving c..")
			x, res, rank, _ = np.linalg.lstsq(A, b, rcond=None)
			print(res, rank, A.shape, nComps*n)
			print("done.")
			CMat = np.zeros((n, nt, nt))
			for i in range(len(xInd_To_CT1T2Q)):
				cInds = xInd_To_CT1T2Q[i]
				CMat[cInds[2], cInds[0], cInds[1]] = x[i]
				CMat[cInds[2], cInds[1], cInds[0]] = x[i]
			c = {(t1,t2):CMat[:, types.index(t1), types.index(t2)] for t1 in types for t2 in types if t1<=t2}
			return c
		else:
			H = np.zeros((n, len(types), len(types)))
			for i in range(len(types)):
				for j in range(len(types)):
					g = gs[(min(types[i], types[j]), max(types[i], types[j]))]
					qs, H[:,i,j] = radialFourierTransformRToQ(rs,  (g - 1)*np.exp(-0*rs**2/(2*200**2)))
			
			lhs = np.eye(len(types))[np.newaxis,:,:] + rhoVec[np.newaxis, np.newaxis, :]*H
			rhs = H
			if basis:
				A = []
				b = []
				Basis = [radialFourierTransformRToQ(rs,  b)[1] for b in basis]
				
				
				nt = len(types)
				xInds =  self.__getBasisXInd(nt, len(basis))
				# eqInd = [(t1,t2, ri) for t1 in range(nt) for t2 in range(nt) for ri in range(n) if t1<=t2]
				nEqs = nt**2*n #nt*(nt+1)//2*n
				A = np.zeros(( nEqs, len(xInds)))
				b = np.zeros((nEqs,))
				weightsByQ = np.ones(qs.shape)
				# weightsByQ = 1/qs
				# weightsByQ = qs**2
				for xi in range(len(xInds)):
					xc = np.zeros((n, nt, nt))
					xc[:, xInds[xi][0], xInds[xi][1]] += Basis[xInds[xi][2]]
					A[:, xi] = (np.einsum("ijk,ikl->ijl",lhs, xc)*weightsByQ[:,np.newaxis,np.newaxis]).flatten()
				b = (rhs*weightsByQ[:,np.newaxis,np.newaxis]).flatten()
				x = np.linalg.lstsq(A, b, rcond=None)[0]
				return x
			else:
				CMat = np.linalg.solve( lhs, rhs)
		
		c = {(t1,t2):radialFourierTransformQToR(qs, CMat[:, types.index(t1), types.index(t2)])[1] for t1 in types for t2 in types if t1<=t2}
		# c = {(t1,t2):CMat[:, types.index(t1), types.index(t2)] for t1 in types for t2 in types if t1<=t2}
		return c
	

	def getGFromC(self, c, dr, n, matrixConv = True):
		if matrixConv:
			dq=np.pi/n/dr
			qs = (0.5+np.arange(n)) * dq
			rs = (0.5+np.arange(n)) * dr

			types = list(self.cells.keys())
			nt = len(types)
			rhos = self.getRhos()
			rhoVec = np.array([rhos[t] for t in types])

			xInd_To_CT1T2Q = [(t1, t2, qi) for t1 in range(nt) for t2 in range(nt) for qi in range(n) if types[t1]<=types[t2]]
			CT1T2Q_To_xInd = {**{xInd_To_CT1T2Q[i]:i for i in range(len(xInd_To_CT1T2Q))}, **{(xInd_To_CT1T2Q[i][1], xInd_To_CT1T2Q[i][0], xInd_To_CT1T2Q[i][2]):i for i in range(len(xInd_To_CT1T2Q))}}

			nComps = nt*(nt+1)//2
			nEqs = nt**2*n
			A = np.zeros(( nEqs, nComps*n))
			b = np.zeros(( nEqs,))
			eqInd = 0
			for t2Eq in range(nt):
				for t1Eq in range(nt):
					for t1C in range(nt):
						xInd = CT1T2Q_To_xInd[(t1C, t2Eq, 0)]
						A[eqInd:eqInd+n, xInd:xInd+n] = - getRadialConvolutionMatrix(dr, n, rhoVec[t1C]*c[(min(types[t1Eq], types[t1C]), max(types[t1Eq], types[t1C]))])
						if t1Eq==t1C:
							A[eqInd:eqInd+n, xInd:xInd+n] += np.eye(n)
						b[eqInd:eqInd+n] = c[(min(types[t1Eq], types[t2Eq]), max(types[t1Eq], types[t2Eq]))]
					eqInd += n

			assert(eqInd == A.shape[0])

			print("Solving h..")
			x, res, _, _ = np.linalg.lstsq(A, b, rcond=None)
			print(res)
			print("done.")

			HMat = np.zeros((n, nt, nt))
			for i in range(len(xInd_To_CT1T2Q)):
				cInds = xInd_To_CT1T2Q[i]
				HMat[cInds[2], cInds[0], cInds[1]] = x[i]
				HMat[cInds[2], cInds[1], cInds[0]] = x[i]
			g = {(t1,t2):(rs, 1 + HMat[:, types.index(t1), types.index(t2)]) for t1 in types for t2 in types if t1<=t2}
			return g
		else:
			dq=np.pi/n/dr
			qs = (0.5+np.arange(n)) * dq
			rs = (0.5+np.arange(n)) * dr

			types = list(self.cells.keys())
			C = np.zeros((rs.shape[0], len(types), len(types)))
			rhos = self.getRhos()
			rhoVec = np.array([rhos[t] for t in types])

			for i in range(len(types)):
				for j in range(len(types)):
					C[:,i,j] = radialFourierTransformRToQ(rs, c[(min(types[i], types[j]), max(types[i], types[j]))])[1]

			HMat = np.linalg.solve( np.eye(len(types))[np.newaxis,:,:] - rhoVec[np.newaxis, np.newaxis, :]*C, C)
			# print("traspose:")
			# print(HMat[0,:,:]-HMat[0,:,:].T)
			# print(HMat[0,:,:]+HMat[0,:,:].T)

			g = {(t1,t2):(rs,1+radialFourierTransformQToR(qs, HMat[:, types.index(t1), types.index(t2)])[1]) for t1 in types for t2 in types if t1<=t2}
		return g
	
	def getHardSphereG(self, sizes, resolution):
		qs = self.getQs(resolution)
		rs = self.getRs(qs)
		types = list(self.cells.keys())
		# c = np.zeros((rs.shape[0], len(types), len(types)))
		# C = np.zeros((rs.shape[0], len(types), len(types)))
		# rhos = self.getRhos()
		# rhoVec = np.array([rhos[t] for t in types])
		c={}
		for t1 in types:
			for t2 in types:
				if t1<=t2:
					c[(t1,t2)] = - 2*np.exp(-rs**2/(2*(sizes[t1]+sizes[t2])**2))
		# c[:, 0, 0] += 0.048*np.exp(-rs**2/(2*(30)**2))

		return self.getGFromC(c, rs[1]-rs[0], len(rs))

	def getSLoss(self, S0, qs, gradient=False):
		res = self.getStructureFactors(qs, gradient=gradient)
		S = res[0]
		L = sum([np.sum((S[(t1,t2)] - S0[(t1,t2)])**2) for t1 in self.cells for t2 in self.cells if t1<=t2])

		if gradient:
			dSdx = res[1]
			dLdx = {k:np.zeros((2,len(self.cells[k]))) for k in self.cells}
			for t1 in self.cells:
				for t2 in self.cells:
					if t1>t2:
						continue
					for ci in range(2):
						dLdx[[t1, t2][ci]] += 2*np.tensordot(dSdx[(t1,t2)][ci], S[(t1,t2)]-S0[(t1,t2)], axes=((2,),(0,)))
			return L, dLdx
		else:
			return L,

	

def generateCellDistribution(counts, S0, W, qs, maxIterations=100, start = None, callback=None, L0=None, startStep=1):
	
	cd = CellDistribution({c:np.random.rand(counts[c], 2)*W for c in counts} if start==None else start, W, torus=True)

	lr = 0.000001
	L=1000
	i=0
	def step(ps, delta, f):
		return {t: np.zeros(0,2) if len(ps[t])==0 else (np.remainder(ps[t] + f*delta[t], W)) for t in ps}

	previousStep = startStep
	while L>1 and i<maxIterations:
		L, dLdx = cd.getSLoss(S0, qs, gradient=True)
		if L0==None: L0=L
		print(f'L/L0: {L/L0*100:.1f}%')

		norm = np.sqrt(np.mean([np.mean(dLdx[t]**2) for t in cd.cells]))
		dLdxNorm = {t:np.transpose(dLdx[t]) / norm for t in cd.cells}

		steps = np.exp(np.linspace(-1., 0.5, 4))*previousStep
		Ls = [CellDistribution(step(cd.cells, dLdxNorm, -s),  W, torus=True).getSLoss(S0, qs)[0] for s in steps]
		# print(Ls)
		bestStep = steps[np.argmin(Ls)]
		print(f"best step: {bestStep:.2g} um")
		previousStep = bestStep
		
		cd = CellDistribution(step(cd.cells, dLdxNorm, -bestStep),  W, torus=True)
		if callback:
			callback(cd)
		i+=1
	return cd, previousStep

if __name__ == "__main__":
	print("Testing radial Fourier transforms..")
	qs = CellDistribution({}, 100).getQs(10)
	fs = np.random.rand(len(qs))
	rs, gs = radialFourierTransformQToR(qs, fs, deWindow = True)
	qs2, f2s = radialFourierTransformRToQ(rs, gs, deWindow = True)
	print(qs)
	print(qs2)
	print(fs)
	print(f2s)

	print("Testing convolution..")
	fs = np.zeros(len(qs))
	fs[len(fs)//2] = 1
	fs/=rs**2*4*np.pi
	sigma = 20
	kern = np.exp(-rs**2/(2*sigma**2))/(2*np.pi*sigma**2)**(3/2)

	_, kernTwice = radialFourierTransformQToR( qs, radialFourierTransformRToQ(rs, kern)[1] * radialFourierTransformRToQ(rs, kern)[1] )
	print("kernTwice:")
	print(kernTwice)
	print("kernTwice expected:")
	kernTwiceExpected = np.exp(-rs**2/(2*2*sigma**2))/(2*np.pi*2*sigma**2)**(3/2)
	print(kernTwiceExpected)

	_, smoothed = radialFourierTransformQToR( qs, radialFourierTransformRToQ(rs, fs)[1] * radialFourierTransformRToQ(rs, kern)[1] )
	print("smoothed:")
	print(smoothed)
	print(np.sum(smoothed*rs**2*4*np.pi))
	

	print("matrix mul convolution:")
	n = len(rs)
	dr = rs[1] - rs[0]
	print(getRadialConvolutionMatrix(dr, n, kern).dot(fs))
	print("... in opposite order:")
	print(getRadialConvolutionMatrix(dr, n, fs).dot(kern))
	
	exit(0)

	W = 300
	n = 100
	ps0 = {f"type{k}":np.random.rand(n, 2)*W for k in range(4)}
	

	ps = {f"type{k}":np.random.rand(n, 2)*W for k in range(4)}
	
	eps = 1e-6
	qs = np.linspace(0.1,1, 10)

	cd0 = CellDistribution(ps0, W, torus=True)
	S0, = cd0.getStructureFactors(qs)

	# generateCellDistribution(cd0.getCounts(), S0, W)

	testFiniteDifference = True
	if testFiniteDifference:
		cd = CellDistribution(ps, W, torus=True)
		L, dLdx = cd.getSLoss(S0, qs, gradient=True)

		for k in ps:
			for di in range(2):
					for i in range(n):
						shiftedPs = np.array(ps[k])
						shiftedPs[i, di] += eps
						finiteDiff = (CellDistribution({ki:shiftedPs if ki==k else ps[ki] for ki in ps}, W, torus=True).getSLoss(S0, qs)[0] - L) / eps
						print(f"finite diff: {finiteDiff}")
						print(f"analytic: {dLdx[k][di,i]}")

	
