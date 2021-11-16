import numpy as np
import matplotlib.pyplot as pl


def rho(t, r):
	ns = np.arange(1,2000)
	return 1 + 2/(np.pi*r)*( (-1)**ns*np.exp(-t*np.pi**2*ns**2)*np.sin(np.pi*ns*r)/ns).sum()


rs = np.linspace(0,1,1000)[1:]

pl.figure(figsize=(4.7,3.6))
for t in [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 1]:
	pl.plot(rs, [rho(t, r) for r in rs], label=f'$t = {t} r_0^2/D$')

pl.xlabel('$r/r_0$')
pl.ylabel(r'$\rho/\rho_0$')
pl.xlim([0, 1])
# pl.ylim([0,1])
pl.legend(loc='center left')
pl.grid()
pl.savefig("out/spheroid_diffusion.pdf")
pl.show()