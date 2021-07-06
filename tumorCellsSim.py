import numpy as np
import matplotlib.pyplot as pl
import scipy
from scipy import signal
import matplotlib.animation as animation
from scipy.spatial import cKDTree
#micrometers
L = 2000

dt = 0.6 #hours

types = ['healthy', 'cancer', 'immune']
colors = {'healthy':(0,1,0), 'cancer':(1,0,0), 'immune':(0,0,1)}
cells = {'healthy':np.random.rand(7000,2)*L, 'cancer':np.array([[L/2, L/2]]), 'immune':np.random.rand(150,2)*L,}

def updatefig(*args):
	for t in cells:
		inside =  (cells[t][:,0] > 0) & (cells[t][:,0] < L) & (cells[t][:,1] > 0) & (cells[t][:,1] < L)
		cells[t] = cells[t][inside, :]
	trees = {t:cKDTree(cells[t]) for t in cells}

	newCells = {t:[] for t in cells}
	
	rad = 30
	displacements = {t: np.zeros(cells[t].shape) for t in cells}

	switchMat = {('cancer', 'immune'):0.2, ('cancer', 'healthy'):0.2, ('healthy','immune'):1.5}
	pairs={}
	for t1 in cells:
		for t2 in cells:
			pairs[(t1,t2)] = trees[t1].query_ball_tree(trees[t2], rad)
			for i in range(len(pairs[(t1,t2)])):
				c = cells[t1][i, :]
				for o in pairs[(t1,t2)][i]:
					diff = c-cells[t2][o,:]
					n = np.sum(diff**2)
					if n>0:
						displacements[t1][i,:] += 5.*diff/n

	for t1 in cells:
		for t2 in cells:
			pairList = np.array( [(i,j) for i,ps in enumerate(pairs[(t1,t2)]) for j in ps ] )
			
			if t1<t2:
				if pairList.shape[0]:
					sPairs = pairList[ np.random.rand(pairList.shape[0])<dt*switchMat[(t1,t2)], : ]
					
					cells[t1][sPairs[:,0],:], cells[t2][sPairs[:,1],:] = cells[t2][sPairs[:,1],:], cells[t1][sPairs[:,0],:]
					displacements[t1][sPairs[:,0],:], displacements[t2][sPairs[:,1],:] = displacements[t2][sPairs[:,1],:], displacements[t1][sPairs[:,0],:]
				
			#pick some switches..



	for t1 in cells:
		if t1 == 'healthy':
			continue
		for ci in range(cells[t1].shape[0]):
			cc = cells[t1][ci, :]
			if t1=='cancer':
				growth = 0.1
			elif t1=='immune':
				if len(trees['cancer'].query_ball_point(cc, 30))>0:
					growth = 0.2
				else:
					growth = 0
			if np.random.rand()<dt*growth:
				newCells[t1].append(ci)
				for t2 in cells:
					diffs = cells[t2] - cc
					ns = np.sum(diffs**2, axis=1)
					
					displacements[t2][ns>0,:] += 100 * diffs[ns>0,:] / ns[ns>0][:,np.newaxis]
			
	for t in cells:
		cells[t] += displacements[t]
		
	for t in cells:
		randDirs = np.random.normal(size=(len(newCells[t]),2))
		randDirs = 0.5*rad*randDirs/np.linalg.norm(randDirs, axis=1)[:, np.newaxis]
		cells[t] = np.concatenate((cells[t], cells[t][newCells[t]] + randDirs), axis=0)
		cells[t][newCells[t]] -= randDirs
	
	for t in cells:
		sc[t].set_offsets(cells[t])
			
	return sc.values()

fig = pl.figure(figsize=(8,8))
sc={}
for t in cells:
	sc[t] = pl.scatter(cells[t][:,0],cells[t][:,1], s=1, color=colors[t])
pl.xlabel("x [μm]")
pl.ylabel("y [μm]")

ani = animation.FuncAnimation(fig, updatefig, interval=10)#, blit=True)

pl.show()
