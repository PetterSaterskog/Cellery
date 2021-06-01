from scipy.fft import dst, dct
import numpy as np
import matplotlib.pyplot as pl

# transform from qs to rs
def radialFourierTransformQToR(qs, fs):
	n = len(qs)
	rs = np.arange(1,n+1)*np.pi/(n+1)/qs[0]
	return rs, dst(fs * qs, type=1) / rs / (2*(n+1))

def radialFourierTransformRToQ(rs, fs):
	n = len(rs)
	qs = np.arange(1,n+1)*np.pi/(n+1)/rs[0]
	return qs, dst(fs * rs, type=1) / qs 

n = 1000
qs = np.arange(1,n+1)
a = np.random.rand(n)

# print(a)
rs, af = radialFourierTransformQToR(qs, a)


for i in [1.5]:
	dists = np.array([0.3,0.4,0.5,0.6, 1.5,1.9])
	af = np.sum( np.sin( dists[:, np.newaxis] * qs[np.newaxis, :] ) * dists[:, np.newaxis], axis=0) / qs * np.exp(-4*qs**2/(qs[-1]**2))

	# print(af)
	rs, a2 = radialFourierTransformQToR(qs, af)

	pl.plot(rs, a2)
pl.show()