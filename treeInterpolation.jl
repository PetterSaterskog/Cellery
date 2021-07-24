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