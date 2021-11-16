import numpy as np
import scipy.special

#rs are right bin edges. First bin starts at 0.
#rho are values in bin centers
def drhodt(rs, rho, V, gamma, D):
	# Neuman BCs
	rhoExtended = np.concatenate([rho[:1, :], rho, rho[-1:, :]], axis=0)
	rhoExtendedUp = np.concatenate([rho, rho[-1:, :]], axis=0)
	edges = np.concatenate([[0], rs], axis=0)
	binCenters = (edges[:-1] + edges[1:])/2
	dx = rs[1]

	#at edges
	gradRho = np.diff(rhoExtended, axis=0) / dx
	
	#at binCenters
	laplaceRho = np.diff(edges[:, None]**2*gradRho, axis=0)/dx / binCenters[:, None]**2

	#at binCenters
	source = gamma[None, :]*rho
	
	# integrated at annuli between all edges
	rs2Int = np.diff(edges**3/3)

	# at rs
	sourceInt = np.cumsum(rs2Int*(V[None, :]*source).sum(axis=1), axis=0)

	# at rs
	u = sourceInt / rs**2 + np.sum(V[None, :]*D[None, :]*np.diff(rhoExtendedUp, axis=0)/dx, axis=1)
	#at edges
	uEdges = np.concatenate([[0], u])

	#div-free advection, at edges
	divFreeAdv = - uEdges[:, None]*gradRho

	# at bincenters
	return source - rho*(V[None, :]*(source+D*laplaceRho)).sum(axis=1)[:, None] + (divFreeAdv[:-1]+divFreeAdv[1:])/2 + D[None, :]*laplaceRho

def gaussianInt3d(r, r0):
	return -((np.sqrt(2/np.pi)*r*np.exp(-r**2/(2*r0**2)))/(r0)) + scipy.special.erf(r/(np.sqrt(2)*r0))

def getTumor(nStartCells, startWidth, D, V, t, L):
	rs = np.linspace(0, L, int(L))[1:]
	ts = np.linspace(0, t, int(500*t))
	dt = ts[1] - ts[0]

	cs = np.zeros((len(rs), len(ts), 2))
	gamma = np.array([0, 1])

	edges = np.concatenate([[0], rs], axis=0)
	vols = np.diff(4*np.pi*edges**3/3)
	cs[:, 0, 1] = np.diff(gaussianInt3d(edges, startWidth))/vols * nStartCells
	cs[:, 0, 0] = (1 - V[1]*cs[:, 0, 1]) / V[0]
	
	for i in range(1, len(ts)):
		cs[:, i, :] = np.clip( cs[:, i-1] + dt * drhodt(rs, cs[:, i-1], V, gamma, D), 0, 1/V)
	return rs, ts, cs

if __name__ == "__main__":
	V = np.array([1,1])
	rs, ts, cs = getTumor(2, 10, np.array([400, 400]), V, 20, 1000)

	edges = np.concatenate([[0], rs], axis=0)
	vols = np.diff(4*np.pi*edges**3/3)
	import matplotlib.pyplot as pl
	for i in range(len(V)):
		pl.figure()
		pl.imshow(np.clip(cs[:, ::len(ts)//len(rs), i].T, 0, 1/V[i]), origin = 'lower')

	# check that the growth is exponential 
	pl.figure()
	pl.plot(ts, np.log( vols.dot(cs[:,:,1]) ), label='log number of cancer cells')
	pl.plot(ts, np.log(2)+ts, label='log(2)*t*gamma')
	pl.legend()

	pl.figure()
	for i in range(len(V)):
		pl.plot(rs, V[i]*cs[:, -1, i])
	pl.plot(rs, cs[:, -1, :].dot(V), ls='--')

	pl.show()