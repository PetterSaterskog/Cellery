import numpy as np
from scipy.signal import convolve, fftconvolve
from scipy import fft

n=1022
a = np.random.rand(*(2*[n])).astype(np.float32)
b = np.random.rand(*(2*[1+2*n])).astype(np.float32)

import time

# with fft.set_workers(-1):
for i in [True]:
	start = time.time()
	if i:
		c=fftconvolve(a,b, mode='same')
	else:
		c=convolve(a,b, mode='same', method='direct')
	end = time.time()
	print(i, c[:2,:2], end - start)
