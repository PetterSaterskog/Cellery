import time
import numpy as np
from scipy.signal import convolve, fftconvolve
import scipy.fftpack
import scipy.fft
import numpy.fft
import pyfftw


n=101
d=3
dtype = np.float32

np.random.seed(0)
a = np.random.normal(size=(d*[n])).astype(dtype)
b = np.random.normal(size=(d*[2*n - 1])).astype(dtype)

pyfftw.config.NUM_THREADS = 64
with scipy.fft.set_workers(-1):
	# with numpy.fft.set_workers(-1):
	for i in range(5):
		start = time.time()
		if i==0:
			c=fftconvolve(a,b, mode='same')
		if i==2:
			fastLen = scipy.fftpack.next_fast_len(2*n - 1)
			shape = d*[fastLen]
			aF = scipy.fft.rfftn(a, s=shape)
			bF = scipy.fft.rfftn(b, s=shape)
			c = scipy.fft.irfftn(aF*bF, s=shape)[d*(np.s_[n-1:2*n-1],)]
		if i==1:
			fastLen = scipy.fftpack.next_fast_len(2*n - 1)
			shape = d*[fastLen]
			aF = numpy.fft.rfftn(a, s=shape)
			bF = numpy.fft.rfftn(b, s=shape)
			c = numpy.fft.irfftn(aF*bF, s=shape)[d*(np.s_[n-1:2*n-1],)]
		if i==3:
			with scipy.fft.set_backend(pyfftw.interfaces.scipy_fft, only=True):
				pyfftw.interfaces.cache.enable()

				fastLen = scipy.fftpack.next_fast_len(2*n - 1)
				shape = d*[fastLen]
				aF = scipy.fft.rfftn(a, s=shape)
				bF = scipy.fft.rfftn(b, s=shape)
				c = scipy.fft.irfftn(aF*bF, s=shape)[d*(np.s_[n-1:2*n-1],)]
		if i==4:
			with scipy.fft.set_backend(pyfftw.interfaces.scipy_fft, only=True):
				pyfftw.interfaces.cache.enable()
				c=fftconvolve(a,b, mode='same')
		end = time.time()
		print(i, c.flatten()[:3], c.flatten()[-3:], c.dtype, end - start)