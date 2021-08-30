import numpy as np
import scipy.optimize
from scipy.spatial import cKDTree
from numba import jit

# some utilities for analysing 2d microscopy slides of 3d cell distributions

# cumulative probability distribution Of DistanceBetweenUniformlySampledPointsIn unit hypercube
# See e.g. https://people.kth.se/~johanph/habc.pdf on how to generalize this to non-square images.
def distanceBetweenUniformlySampledPointsInUnitHypercubeCDF(r, d):
	if d==2:
		if r<0:
			return 0
		if r<1:
			return np.pi*r**2-8*r**3/3+r**4/2
		if r<np.sqrt(2):
			return  1/3 - 2 * r**2 - np.pi*r**2 - r**4/2 + 4*np.sqrt(r**2 - 1) + 8/3*(r**2-1)**(3/2) + 4*r**2*np.arcsin(1/r)
		return 1
	if d==3:
		if r<0:
			return 0
		if r < 1:
			return ((48 - 5*r)*r**5 - 5*np.pi*r**3*(-8 + 9*r))/30
		if r < np.sqrt(2):
			return (43 - 5*np.pi)/30 + (-4 - np.pi)/3 + (-80*np.pi*r**3 + 10*r**6 + 24*np.sqrt(-1 + r**2) + r**4*(45 - 96*np.sqrt(-1 + r**2)) - 3*r**2*(5 - 30*np.pi + 36*np.sqrt(-1 + r**2)) + 180*r**4*np.arctan(np.sqrt(-1 + r**2)))/30.
		if r < np.sqrt(3):
			return (43 - 5*np.pi)/30 + (-4 - np.pi)/3 + (-100 - 6*(41 - 30*np.pi) + 180*np.pi - 160*np.sqrt(2)*np.pi)/30 + (37 - (33 - 16*np.sqrt(2))*np.pi)/3 + ((-5 + 6*np.pi)*r**2)/2 - (8*np.pi*r**3)/3 + (3*(-1 + np.pi)*r**4)/2 - r**6/6 + (2*np.sqrt(-2 + r**2)*(1 + 9*r**2 + 4*r**4))/5 + np.arctan((-2 + r)/np.sqrt(-2 + r**2)) - np.arctan((2 + r)/np.sqrt(-2 + r**2)) -  6*r**2*(2 + r**2)*np.arctan(np.sqrt(-2 + r**2)) + 8*r**3*np.arctan(r*np.sqrt(-2 + r**2))
		return 1


@jit(nopython=True)
def f(r, R, s):
	if R > r:
		return -np.pi*R**2
	if R**2 < r**2 - s**2:
		return (s**2 - 6*r**2)*np.pi/6
	return (-2*np.pi*((s**2*R**2)/2 + (2*s*(r**2 - R**2)**(3/2))/3 - (r**2 - R**2)**2/4))/s**2

# This returns a matrix M, such that: 2d radial distribution function = M * 2d radial distribution function + asym
# Both radial distribution functions are represented as vectors of particles densities in bins. Each bin is assumed to have uniform density in space (not (?) true also for 2d bins, can be fixed).
# asym is the contribution from assuming that the 2d rdf = 1 past the last edge2d

def get3dToSliceRDF(edges2d, edges3d, thickness):
	prim = [[f(r, R, thickness) for r in edges3d] for R in edges2d]
	areas = np.diff(np.pi*edges2d**2)
	M = np.diff(np.diff(prim, axis=0), axis=1) / areas[:, np.newaxis]
	asym = - np.diff(prim, axis=0)[:, -1] / areas
	return M, asym

# Assume cells are spheres, and they are detected if the projected radius of their union with the slice geometry is larger than detectionRadius
# Further 
def getVolNumberDensityFromAreaNumberDensity(volNumDensity, sliceThickness, detectionRadius):

	pass

@jit(nopython=True)
def getHardShellExample(d, n, L, r=0.05):
	ps = np.random.rand(n, d)*L
	for i in range(30*n):
		dE = 0
		i1 = np.random.randint(n)
		p1=ps[i1]
		p1n = np.random.rand(d)*L
		for i2 in range(n):
			p2 = ps[i2]
			diff = p1-p2
			diff = np.remainder(diff + L/2, L) - L/2
			if diff.dot(diff) < 4*r*r:
				dE -= (2*r - np.linalg.norm(diff))**2
			diff = p1n-p2
			diff = np.remainder(diff + L/2, L) - L/2
			if diff.dot(diff) < 4*r*r:
				dE += (2*r - np.linalg.norm(diff))**2
		if dE < 0:
			ps[i1] = p1n
	return ps

def getHardShellExample2(d, n, L, r=0.05):
	ps = np.random.rand(n, d)*L
	for i in range(10):
		pairs = cKDTree(ps, boxsize=(L,)*d).query_pairs(2*r, output_type="ndarray")
		diffs = np.remainder(ps[pairs[:,1],:] - ps[pairs[:,0],:] + L/2, L) - L/2
		diffNorms = np.linalg.norm(diffs, axis=1)
		f = 0.1 * (2*r-diffNorms)[:, np.newaxis]*diffs/diffNorms[:, np.newaxis]
		for i in range(d):
			ps[:,i] += np.bincount(pairs[:,1], weights=f[:,i], minlength=n) - np.bincount(pairs[:,0], weights=f[:,i], minlength=n)
			ps = np.remainder(ps, L)
	return ps

def getG(ps, edges, L):
		assert( edges[-1] <= L*np.sqrt(2) )
		genAreas = [distanceBetweenUniformlySampledPointsInUnitHypercubeCDF(e/L, ps.shape[1]) for e in edges]
		uniformProbabilitiesPerBin = np.diff(genAreas)
		pairs = cKDTree(ps).query_pairs(edges[-1], output_type = 'ndarray')
		norm = 2 / uniformProbabilitiesPerBin / len(ps) / (len(ps)-1)
		dists = np.linalg.norm(ps[pairs[:,0]]-ps[pairs[:,1]], axis=1)
		hist = np.histogram(dists, edges)[0]
		return hist * norm

def getS():

def getMinThickness():
	pass

def plotHist(edges, counts, ls='-', label=''):
	pl.ylim([0,2])
	pl.plot(np.repeat(edges, 2)[1:-1], np.repeat(counts, 2), ls, label=label)

if __name__ == "__main__":
	import matplotlib.pyplot as pl

	L=1200
	# edges3d = np.linspace(0, 300, 100)
	edges3d = np.linspace(0, 100, 100)
	edges2d = np.linspace(0, 100, 200)

	if True:
		ps = getHardShellExample2(3, int((L/20)**3), L, 10)
		np.savetxt(f"out/hardShellCache3.csv", ps)
	else:
		print("Loading cells..")
		ps = np.loadtxt(f"out/hardShellCache2.csv")
	print("Measuring rdf3d..")
	rdf3d = getG(ps, edges3d, L)

	# pl.plot(slicePs[:,0],slicePs[:,1],'.')
	import tqdm
	pl.figure()
	for thickness in tqdm.tqdm([5, 10, 15, 20]):
		rdf2ds = []
		for z in np.linspace(thickness/2, L-thickness/2, int(L/thickness)):
			slicePs = ps[np.abs(ps[:, 2]-z) < thickness/2][:, :2]
			rdf2ds.append(getG(slicePs, edges2d, L))
		rdf2d = np.average(rdf2ds, axis=0)

		testThicknesses = np.linspace(0.01, 35, 100)
		residues = []
		regs = []
		for testThickness in testThicknesses:
			M, asym = get3dToSliceRDF(edges2d, edges3d, testThickness)
			rdf2dFrom3d = M.dot(rdf3d)
			# weights = np.ones(edges2d.shape[0]-1)
			weights = np.diff(np.pi*edges2d**2)
			# rdf3dFrom2d, res, _, _ = np.linalg.lstsq(M, rdf2d, rcond=None)
			rdf3dFrom2d, res = scipy.optimize.nnls(weights[:, np.newaxis]*M, weights*(rdf2d-asym))
			err = (weights*(M.dot(rdf3dFrom2d) + asym - rdf2d))
			res = err.dot(err)
			residues.append(res)
			# regs.append(np.sum((edges3d[1:-1]**2*np.diff(rdf3dFrom2d))[:-5]**2))
		# rdf3dFrom2d = np.linalg.solve(M, rdf2d)
		
		pl.plot(testThicknesses, residues, label=f'{thickness} um')
		am = np.argmin(residues)
		pl.plot([testThicknesses[am]],[residues[am]],'x',color='red')
	# pl.plot(ratios, regs/np.max(regs))
	pl.legend()
	pl.grid()

	thickness = 25
	rdf2ds = []
	for z in np.linspace(thickness/2, L-thickness/2, int(L/thickness)):
		slicePs = ps[np.abs(ps[:, 2]-z) < thickness/2][:, :2]
		rdf2ds.append(getG(slicePs, edges2d, L))
	rdf2d = np.average(rdf2ds, axis=0)

	M, asym = get3dToSliceRDF(edges2d, edges3d, thickness)
	rdf2dFrom3d = M.dot(rdf3d) + asym
	rdf3dFrom2d, res = scipy.optimize.nnls(M, rdf2d - asym)

	pl.figure()
	plotHist(edges3d, rdf3d, ls='-', label="3d RDF")
	plotHist(edges3d, rdf3dFrom2d, ls='--', label='3d RDF from 2d')
	plotHist(edges2d, rdf2d, ls='-', label="2d RDF from slice")
	plotHist(edges2d, rdf2dFrom3d, ls='--', label="2d RDF from 3d")
	pl.legend()
	# pl.figure()
	# rs = np.linspace(0,2,100)
	# pl.plot(rs, [distanceBetweenUniformlySampledPointsInUnitHypercubeCDF(r, 2) for r in rs])
	# pl.plot(rs, [distanceBetweenUniformlySampledPointsInUnitHypercubeCDF(r, 3) for r in rs])

	# pl.savefig("out/deslicing/test.pdf")

	pl.show()
