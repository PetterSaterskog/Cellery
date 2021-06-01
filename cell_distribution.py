import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix
from scipy.fft import dst, dct, next_fast_len
from multiprocessing import Pool

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
	pair, cells, trees, qs, W, dz, torus, gradient = arg
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

	if torus:
		weights = np.ones(dists.shape)
	else:
		assert(gradient==False) #not handling derivative of this..
		weights = np.array([(2*np.pi*(d/W))/g(d/W) for d in dists])

		# ddistdx[ti][di][cell, dist]
	# l2 =  4/qs[0]**2
	# dqrdx = [ [ddistdx[ci][di][:, np.newaxis] * qs[np.newaxis, :] for di in range(2)] for ci in range(2)]
	dq = qs[1]-qs[0]
	batchSize = 10000 #batch this sum to not run out fo memory
	S = 0
	for i in range(1+dists.shape[0]//batchSize):
		S += np.sum( weights[i*batchSize:(i+1)*batchSize, np.newaxis]*np.sin( dists[i*batchSize:(i+1)*batchSize, np.newaxis] * qs[np.newaxis, :] )*np.exp(-dists[i*batchSize:(i+1)*batchSize, np.newaxis]**2/(2/dq**2)), axis=0)
	S *=  2 * np.exp(-1*qs**2/(qs[-1]**2)) / qs / dz
	# S = np.sum( weights[:, np.newaxis]*np.sin( dists[:, np.newaxis] * qs[np.newaxis, :] )*np.exp(-dists[:, np.newaxis]**2/(2*l2)) / qs[np.newaxis, :] / dists[:, np.newaxis], axis=0)
	if gradient:
		ddistdx = [ [csr_matrix(([-1,1][ci]*diff[:,di]/dists, (cellPairs[:, ci], np.arange(len(dists)) )), shape=(len(cells[pair[ci]]), len(dists))) for di in range(2)] for ci in range(2)]
		return S, [ [ ddistdx[ci][di].dot(np.cos( dists[:, np.newaxis] * qs[np.newaxis, :] )*np.exp(-dists[:, np.newaxis]**2/(2*l2)) / dists[:, np.newaxis]
                    -  np.sin( dists[:, np.newaxis] * qs[np.newaxis, :] )*np.exp(-dists[:, np.newaxis]**2/(2*l2)) / qs[np.newaxis, :]/dists[:, np.newaxis]**2
		    - np.sin( dists[:, np.newaxis] * qs[np.newaxis, :] )*np.exp(-dists[:, np.newaxis]**2/(2*l2)) / qs[np.newaxis, :] /l2)  for di in range(2)] for ci in range(2)]
	else:
		return S,

# transform from qs to rs
def radialFourierTransformQToR(qs, fs, deWindow = False):
	n = len(qs)
	dq = qs[1]-qs[0]
	rs = (0.5+np.arange(n))*np.pi/n/dq
	return rs, dst(fs * qs, type=4)*(np.exp(rs*rs*dq**2/2) if deWindow else 1) / rs / (4*np.pi*n) / (rs[1]-rs[0])

def radialFourierTransformRToQ(rs, fs):
	n = len(rs)
	qs = np.arange(1,n+1)*np.pi/(n+1)/rs[0]
	return qs, dst(fs * rs, type=1) / qs 
	
class CellDistribution():
	def __init__(self, cells, W, z = 1, torus = False):
		self.cells = {t:np.array(cells[t]) for t in cells}
		self.W, self.z = W, z
		self.torus = torus
		self.trees = {t:cKDTree(cells[t], boxsize = (W,W) if torus else None) for t in cells}

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
		pairs = [(t1, t2) for t1 in self.trees for t2 in self.trees if t1<=t2 and not (t1==t2 and len(self.cells[t1])==0)]
		with Pool(nThreads) as pool:
			res = list(zip(*pool.map(getPairGrad, [(p, self.cells, self.trees, qs, self.W, self.z, self.torus, gradient) for p in pairs])))
		
		if gradient:
			S = dict(zip(pairs, res[0]))
			dSdx = dict(zip(pairs, res[1]))
			return S, dSdx
		else:
			return [dict(zip(pairs, res[0]))]
	
	def getRadialDistributionFunction(self, resolution):
		maxr = self.W * np.sqrt(2) #*2, <- works fine with larger values too, though its slower. Perhaps good for less aliasing?
		n = next_fast_len(int(2*maxr/resolution+1))
		qs = (0.5+np.arange(n)) * np.pi / maxr
		print("Calculating structure factor...")
		S, = self.getStructureFactors(qs)
		counts = self.getCounts()
		rhos = self.getRhos()
		Sn = {p:S[p]/(counts[p[0]]*rhos[p[1]]) for p in S}

		print("Preparing matrices...")
		types = list(self.cells.keys())
		rhoVec = np.array([rhos[t] for t in types])
		Smat = np.array([[[Sn[(min(t1,t2), max(t1, t2))][i] for t2 in types] for t1 in types] for i in range(len(qs))])
		
		print("Solving...")
		DMat = np.linalg.solve( np.eye(len(types))[np.newaxis,:,:] + rhoVec[np.newaxis, np.newaxis, :]*Smat, Smat)
		# DMat = np.linalg.solve( np.eye(len(types))[np.newaxis,:,:] , Smat)
		
		g = {p:radialFourierTransformQToR(qs, Sn[p], deWindow = True) for p in S}
		C = {p:radialFourierTransformQToR(qs, DMat[:, types.index(p[0]), types.index(p[1])], deWindow = True) for p in S}
		return g, C

	def getDirectCorrelationFunction(self, rs):
		resolution = rs[0] #um
		qs = np.linspace((2*np.pi)/(self.W/2), 2*np.pi / resolution, 100)
		
		counts = self.getCounts()
		rhos = self.getRhos()
		Sm1 = self.getStructureFactors(qs)
		
		types = self.cells.keys()

		rhoVec = np.array([rhos[t] for t in types])
		Smat = np.array([[[types[(min(t1,t2), max(t1,t2))][i] for t2 in types] for t1 in types] for i in range(len(qs))])

		cMat = np.linalg.solve(np.eye(len(types))[np.newaxis,:,:] + rhos[np.newaxis, np.newaxis, :]*Smat, Smat)

		# fG = 
		# in 3d:
		# S(q) = 1 + 4 pi rho int_0^inf dr r (g(r) - 1) sin(qr) / q

		h = {p:g[p]-1 for p in g}


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

	
