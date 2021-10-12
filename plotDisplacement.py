

import numpy as np
import matplotlib.pyplot as pl

L = 30
xs = np.linspace(-L, L,1000)

cells = np.random.rand(40,2)*2*L-L
print(cells)

def transform(x, y, r0):
	r = np.sqrt(x**2 + y**2)
	nr = (r**2 + r0**2)**(1/2)
	theta = np.arctan2(x, y)
	return nr*np.sin(theta), nr*np.cos(theta)

for r0 in [0,8]:
	pl.figure()
	
	for y in np.linspace(-L, L,20):
		xs2, ys2 = transform(xs, y, r0)

		pl.plot(xs2, ys2, color='gray')
		pl.plot(ys2, xs2, color='gray')
	thetas = np.linspace(0,2*np.pi,1000)
	pl.plot(r0*np.sin(thetas), r0*np.cos(thetas), color='red', linestyle='--')

	for i in range(cells.shape[0]):
		cx,cy=transform(cells[i,0], cells[i,1], r0)
		pl.plot([cx],[cy], marker='x')
		print(cx, cy)
	# (kernelCoords*(((1 + (1 - np.exp(-(kernelRs/maxR)**d))*(maxR / kernelRsReg)**d )**(1/d) - 1)/(maxR**d*sphereVol(d)))[..., newaxis]).
	pl.axis('square')
	pl.xlim([-L,L])
	pl.ylim([-L,L])
	pl.xlabel('$x$ [$\mu$m]')
	pl.ylabel('$y$ [$\mu$m]')
	pl.savefig(f'out/displacement_r0={r0}.pdf')
pl.show()