import numpy as np

xs = np.linspace(0,1000,1000)
ts = np.linspace(0, 2, 60000)

dx = xs[1] - xs[0]
dt = ts[1] - ts[0]

c0 = 1

def dcdt(c, d=200):
	intC = np.cumsum(c) - c/2
	cExtended = np.concatenate([[c0], c, [0]])
	dcdx = np.diff(cExtended[:-1] + cExtended[1:]) / dx / 2
	d2cd2x = np.diff(np.diff(cExtended)) / dx**2
	dh = 200
	dc = 50
	d = (1-c**6)*2000 + 20
	return c - c**2 - intC * dcdx + d * d2cd2x/100

cs = np.zeros((len(xs), len(ts)))
cs[:100, 0] = c0

for i in range(1, len(ts)):
	cs[:, i] = np.clip( cs[:, i-1] + dt * dcdt(cs[:, i-1]) ,0,1)

import matplotlib.pyplot as pl

pl.imshow(np.clip(cs[:, ::100].T,0,1), origin = 'lower')

pl.figure()
pl.plot(xs, cs[:, 400*100])
pl.show()