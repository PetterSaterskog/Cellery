import numpy as np

class TreeInterpolation():
	def __init__(self, xMin, xMax, minSize):
		self.xMin, self.xMax = np.array(xMin), np.array(xMax)
		self.center = (self.xMin+self.xMax)/2
		self.sSize = np.sum((self.xMax-self.xMin)**2)
		
		d = len(xMin)
		if self.sSize > (2*minSize)**2:
			self.children = []
			self.childVec = np.array([2**i for i in range(d)])
			childSize = (self.xMax-self.xMin)/2
			c = d*[False]
			while True:
				p = self.xMin + c*childSize
				self.children.append(TreeInterpolation(p, p+childSize, minSize))
				i = 0
				while c[i]:
					c[i] = False
					i+=1
					if i==d:
						break
				else:
					c[i] = True
					continue
				break
		else:
			self.children = []
	
	def reset(self):
		self.v = 0
		for c in self.children:	c.reset()
	
	def add(self, f, p, ratio2=1e-2):
		diff = self.center[:, np.newaxis] - p
		if self.children:
			near = np.sum(diff*diff, axis=0)*ratio2 < self.sSize
			if np.any(near):
				for c in self.children:
					c.add(f, p[:, near], ratio2=ratio2)
			self.v += np.sum(f(diff[:,~near]), axis=-1)
		else:
			self.v += np.sum(f(diff), axis=-1)

	def __call__(self, p):
		assert(np.all(p >= self.xMin) and np.all(p <= self.xMax) )
		if self.children:
			return self.v + self.children[np.sum(self.childVec[p>self.center])](p)
		else:
			return self.v

# class GridInterpolation():
# 	def __init__(self, xMin, xMax, minSize):
# 		self.grids = [np.zeros(d*[2**i]) for i in range()]
# 		self.xMin, self.xMax = np.array(xMin), np.array(xMax)
# 		self.center = (self.xMin+self.xMax)/2
# 		self.sSize = np.sum((self.xMax-self.xMin)**2)
		
# 		d = len(xMin)
# 		if self.sSize > (2*minSize)**2:
# 			self.children = []
# 			self.childVec = np.array([2**i for i in range(d)])
# 			childSize = (self.xMax-self.xMin)/2
# 			c = d*[False]
# 			while True:
# 				p = self.xMin + c*childSize
# 				self.children.append(TreeInterpolation(p, p+childSize, minSize))
# 				i = 0
# 				while c[i]:
# 					c[i] = False
# 					i+=1
# 					if i==d:
# 						break
# 				else:
# 					c[i] = True
# 					continue
# 				break
# 		else:
# 			self.children = []
	
# 	def reset(self):
# 		self.v = 0
# 		for c in self.children:	c.reset()
	
# 	def add(self, f, p, ratio2=1e-2):
# 		for grid

# 		diff = self.center[:, np.newaxis] - p
# 		if self.children:
# 			near = np.sum(diff*diff, axis=0)*ratio2 < self.sSize
# 			if np.any(near):
# 				for c in self.children:
# 					c.add(f, p[:, near], ratio2=ratio2)
# 			self.v += np.sum(f(diff[:,~near]), axis=-1)
# 		else:
# 			self.v += np.sum(f(diff), axis=-1)

# 	def __call__(self, p):
# 		assert(np.all(p >= self.xMin) and np.all(p <= self.xMax) )
# 		if self.children:
# 			return self.v + self.children[np.sum(self.childVec[p>self.center])](p)
# 		else:
# 			return self.v

if __name__ == "__main__":
	import matplotlib.pyplot as pl
	import matplotlib.animation as animation
	from tqdm import tqdm

	d=3
	L = 20
	nFrames = 200
	eps = np.float32(0.2)
	
	ti = TreeInterpolation(d*[-L], d*[L], eps)
	
	np.random.seed(0)
	def galaxy(p, v, r, w, n):
		ps = r*np.random.normal(size=(n,d))
		vs = np.cross(ps, w)
		return p+ps, v+vs
	
	gs = [galaxy(*a) for a in [([-2,-1,0], [0.1,0,0], .3, [0,0,.2], 1000), ([1,1,0], [-0.05,0,0], .4, [0,0,.2], 2000)]]
	ps = np.concatenate([g[0] for g in gs]).astype(np.float32)
	vs = np.concatenate([g[1] for g in gs]).astype(np.float32)

	fig = pl.figure()
	sc = pl.scatter(ps[:,0], ps[:,1], s=2)
	pl.xlim([-L,L])
	pl.ylim([-L,L])
	pl.axis('square')
	dt = np.float32(1e-1)
	tracks = np.zeros((nFrames, ps.shape[0], d), np.float32)
	for i in tqdm(range(nFrames)):
		ti.reset()
		# for p in ps:
		ti.add(lambda diff: -diff/(np.linalg.norm(diff, axis=0) + eps)**3, ps.T, ratio2=1e-1)
		for j in range(ps.shape[0]):
			f = ti(ps[j,:]) / ps.shape[0]
			vs[j,:] += dt*f
		ps+=dt*vs
		inside = np.all(np.abs(ps)<L, axis=1)
		ps = ps[inside]
		vs = vs[inside]
		tracks[i,:ps.shape[0],:] = ps
		tracks[i,ps.shape[0]:,:] = d*[L]
	
	anim = animation.FuncAnimation(fig, lambda i: sc.set_offsets(tracks[i, :, :2]), frames=nFrames)
	writervideo = animation.FFMpegWriter(fps=30) 
	anim.save(f"out/galaxies.avi", writer=writervideo)
	pl.show()