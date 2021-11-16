import numpy as np

xs = np.linspace(0,1000,1000)
ts = np.linspace(0, 2, 60000)

dx = xs[1] - xs[0]
dt = ts[1] - ts[0]

c0 = [1,0]
c1 = [0,0.1]

gamma = 1
vmg = 100

def dcdt(c):
	# cExtended = np.concatenate([[c0], c, [c1]], axis=0)
	cExtended = np.concatenate([c[:1, :], c, [c1]], axis=0)
	dcdx = np.diff(cExtended[:-1] + cExtended[1:], axis=0) / dx / 2
	d2cd2x = np.diff(np.diff(cExtended, axis=0), axis=0) / dx**2
	
	source = np.zeros(c.shape)
	source[:, 0] += gamma*c[:, 0]
	sourceSum = source.sum(axis=1)
	vECM = gamma*(np.cumsum(sourceSum) - sourceSum/2)
	dvECMdx = sourceSum

	v = np.array([vECM,vECM -vmg*(1.2 - c[:, 0])]).T
	dvdx = np.array([dvECMdx, dvECMdx +vmg*dcdx[:, 0]]).T
	
	return -v*dcdx - dvdx*c + source + 500*d2cd2x

cs = np.zeros((len(xs), len(ts), 2))
cs[:100, 0, :] = c0
cs[100:, 0, :] = c1

for i in range(1, len(ts)):
	cs[:, i, :] = np.clip( cs[:, i-1] + dt * dcdt(cs[:, i-1]), 0, 1)

import matplotlib.pyplot as pl
pl.figure()
pl.imshow(np.clip(cs[:, ::100, 0].T, 0, 1), origin = 'lower')
pl.figure()
pl.imshow(np.clip(cs[:, ::100, 1].T, 0, 1), origin = 'lower')
pl.figure()
pl.plot(xs, cs[:, 400*100, 0])
pl.plot(xs, cs[:, 400*100, 1])
pl.show()