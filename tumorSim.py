import numpy as np
import matplotlib.pyplot as pl
import scipy
from scipy import signal
import matplotlib.animation as animation

#micrometers
xs = np.linspace(0, 3000, 200)
n = len(xs)

#days
ts = np.linspace(0, 60, 10)

dx = xs[1] - xs[0]
dt = .1

c0 = 1


# calculate Greens function for stokes flow without 0 flow at inf bc
Gxs = np.linspace(-xs[-1], xs[-1], 2*len(xs) - 1)
GxsM = Gxs[:, np.newaxis]*np.ones(2*[len(Gxs)])
Gps = np.array([GxsM, GxsM.T])
Gden = np.sum(Gps**2, axis=0)
Gden[len(xs)-1, len(xs)-1] = 1 #avoid divide by 0
G = dx*dx*Gps / Gden #multiply by dx*dx because integral is a discrete sum

# comp, x, y
def divergence(f):
	return (f[0, 2:, 1:-1] - f[0, :-2, 1:-1] + f[1, 1:-1, 2:] - f[1, 1:-1, :-2])  / (2*dx)

def pad(f, val):
	return np.pad(f, pad_width=1, mode='constant', constant_values=val)

def gradient(f):
	return np.array([f[:, 2:, 1:-1] - f[:, :-2, 1:-1], f[:, 1:-1, 2:] - f[:, 1:-1, :-2]])  / (2*dx)

def laplacian(f):
	return (f[:, 2:, 1:-1] + f[:, :-2, 1:-1] + f[:, 1:-1, 2:] + f[:, 1:-1, :-2] - 4*f[:, 1:-1, 1:-1])  / dx**2

# y = [healthy, cancer, immune]
def dydt(y, G):
	growth = 0.02
	diffusion = 0. #um^2/s

	bc = [1, 0, 0]
	yP = np.array([pad(y[i, :, :], bc[i]) for i in range(3)])
	f = np.array([np.zeros((n,n)), growth*y[1, : , :], 10*growth*y[2, : , :]*y[1, : , :]])

	divV = np.sum(f, axis = 0 )
	
	v = dx*dx*np.array([scipy.signal.convolve(divV, G[i], mode='same') for i in range(2)])
	# v[0,:,:] += 100

	gradY = gradient(yP)
	# lapY = laplacian(yP)
	# pl.figure()
	# pl.imshow(v[0,:,:], origin = 'lower')
	# pl.show()
	# exit(0)
	
	return -np.sum(gradY * v[:, np.newaxis, :, :], axis=0)  + (f - divV[np.newaxis, :, :] * y) # + diffusion * lapY + f

# pl.imshow(divergence(G), origin = 'lower')

# pl.show()
# exit(0)
# vGreensFun = 

ps = np.stack((xs[:, np.newaxis]*np.ones((n,n)), xs[np.newaxis, :]*np.ones((n,n))), axis=0)
center = [1000,1000]
t = 0
y = np.zeros((3, n, n))
y[1, :, :] += np.exp(-np.sum((ps-np.array([1000,1000])[:,np.newaxis,np.newaxis])**2, axis = 0)/100**2)
y[1, :, :] += np.exp(-np.sum((ps-np.array([1500,1200])[:,np.newaxis,np.newaxis])**2, axis = 0)/50**2)
y[0, :, :] = 1 - y[1, :, :]

# for it in ts:
# 	while t<it:
# 		y = np.clip( y + dt * dydt(y, G),0 , 1)
# 		t+=dt
# 	pl.figure()
# 	pl.imshow(y[1, :, :].T, origin = 'lower', vmin=0, vmax=1)
def colTransf(y):
	colMat = [[0,1,0],[1,0,0],[0,0,1]]
	return y.transpose((2,1,0)).dot(colMat)

def updatefig(*args):
	global y
	growth = 0.02

	f = np.array([
		0.03*y[0, : , :]*y[2, : , :], 
		growth*y[1, : , :]*(1-0.5*y[2, : , :]),
		0.4*(y[2, : , :]*y[1, : , :])**2 - 0.03*y[0, : , :]*y[2, : , :]])

	# instab = 0.2*(y-1/3)
	# f += instab

	divV = np.sum(f, axis = 0 )
	
	v = np.array([scipy.signal.convolve(divV, G[i], mode='same') for i in range(2)])
	
	bc = [1, 0, 0]
	yP = np.array([pad(y[i, :, :], bc[i]) for i in range(3)])
	yP = (yP-0.5)
	yP = 1/(1+np.exp(-10.*yP))
	yNew = y + dt*( f - divV[np.newaxis, :, :] * y )
	yNew[2:3,:,:] += - 10.*laplacian(yP)[2:3,:,:]

	yNew = scipy.interpolate.interpn((xs, xs),
		yNew.transpose((1,2,0)),
		(ps - dt*v).reshape(2, -1).T,
		bounds_error=True ).reshape(n,n,3).transpose((2,0,1))
	
	# yNew = yAdvected 
	
	newImmune = np.random.rand(n, n)<yNew[0,:,:]*dt*0.001
	yNew[0,:,:][newImmune] = 0
	yNew[1,:,:][newImmune] = 0
	yNew[2,:,:][newImmune] = 1

	y = np.clip( yNew, 0, 1)

	im.set_array(colTransf(y))
	leg = pl.gca().legend(
		[pl.Rectangle((0,0),1,1,color=(0,1,0)),
		pl.Rectangle((0,0),1,1,color=(1,0,0)),
		pl.Rectangle((0,0),1,1,color=(0,0,1))],
		["Healthy", "Cancer", "Immune"], loc='upper left',prop={'size':10})
	return im,

fig = pl.figure()
# pl.plot(0,0,c=(1,0,0),label="Cancer")
# leg = pl.gca().legend([pl.Rectangle((0,0),1,1)],["step0"], loc='upper left',prop={'size':12})
im = pl.imshow(colTransf(y), animated=True, origin = 'lower', vmin=0, vmax=1, extent=[0,xs[-1],0,xs[-1]])
pl.xlabel("x [μm]")
pl.ylabel("y [μm]")

ani = animation.FuncAnimation(fig, updatefig, interval=50)#, blit=True)

pl.show()
