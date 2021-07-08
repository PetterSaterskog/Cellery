import numpy as np
import matplotlib.pyplot as pl
import scipy
from scipy import signal
import matplotlib.animation as animation
from scipy.spatial import cKDTree
#micrometers
L = 5000

dt = 0.05 #hours

cellRad = 10 #this assumes no gap, just defined through volPerCell
volPerCell = cellRad**2*np.pi
immuneFraction = 0.03

nCells = L**2 / volPerCell

types = ['healthy', 'cancer', 'immune']
colors = {'healthy':(0,1,0), 'cancer':(1,0,0), 'immune':(0,0,1)}
cells = {'healthy':np.random.rand(int(nCells*(1-immuneFraction)),2)*L, 'cancer':np.array([[L/2, L/2]]), 'immune':np.random.rand(int(nCells*immuneFraction),2)*L,}
switchMat = {('cancer', 'immune'):0.0, ('cancer', 'healthy'):0.5, ('healthy','immune'):1.5}
cancerGrowth = 0.03
immuneGrowth = 0.04

frames = 0
def updatefig(*args):
	for t in cells:
		inside =  (cells[t][:,0] > 0) & (cells[t][:,0] < L) & (cells[t][:,1] > 0) & (cells[t][:,1] < L)
		cells[t] = cells[t][inside, :]
	trees = {t:cKDTree(cells[t]) for t in cells}

	newCells = {t:[] for t in cells}
	
	rad = 30
	displacements = {t: np.zeros(cells[t].shape) for t in cells}

	pairs={}
	for t1 in cells:
		for t2 in cells:
			pairs[(t1,t2)] = trees[t1].query_ball_tree(trees[t2], rad)
			# for i in range(len(pairs[(t1,t2)])):
			# 	c = cells[t1][i, :]
			# 	for o in pairs[(t1,t2)][i]:
			# 		diff = c-cells[t2][o,:]
			# 		n = np.sum(diff**2)
			# 		if n>0:
			# 			displacements[t1][i,:] += 5.*diff/n

	for t1 in cells:
		for t2 in cells:
			if t1<t2:
				ps = pairs[(t1,t2)]
				for i in range(len(ps)):
					if np.random.rand() < dt*switchMat[(t1,t2)]*len(ps[i]):
						j = np.random.randint(len(ps[i]))
						cells[t1][i, :], cells[t2][ps[i][j], :] = np.array(cells[t2][ps[i][j], :]), np.array(cells[t1][i, :])
			# pairList = np.array( [(i,j) for i,ps in enumerate(pairs[(t1,t2)]) for j in ps ] )
			# switchCell1 = np.arange(len(cells))[]
			# if t1<t2:
			# 	if pairList.shape[0]:
			# 		sPairs = pairList[ np.random.rand(pairList.shape[0])<dt*switchMat[(t1,t2)], : ]
					
			# 		cells[t1][sPairs[:,0],:], cells[t2][sPairs[:,1],:] = cells[t2][sPairs[:,1],:], cells[t1][sPairs[:,0],:]
					# displacements[t1][sPairs[:,0],:], displacements[t2][sPairs[:,1],:] = displacements[t2][sPairs[:,1],:], displacements[t1][sPairs[:,0],:]
				
			#pick some switches..

	splitRad = cellRad/2

	for t1 in cells:
		if t1 == 'healthy':
			continue
		for ci in range(cells[t1].shape[0]):
			cc = cells[t1][ci, :]
			if t1=='cancer':
				growth = cancerGrowth
			elif t1=='immune':
				if len(trees['cancer'].query_ball_point(cc, 3*cellRad))>0:
					growth = immuneGrowth
				else:
					growth = 0
			if np.random.rand()<dt*growth:
				newCells[t1].append(ci)
				for t2 in cells:
					diffs = cells[t2] - cc
					ns = np.sum(diffs**2, axis=1)
					
					displacements[t2][ns>0,:] += diffs[ns>0,:] * (np.sqrt(cellRad**2/ns[ns>0] + 1) - 1)[:,np.newaxis] # diffs[ns>0,:] / ns[ns>0][:,np.newaxis]
			
	for t in cells:
		cells[t] += displacements[t]
	
	
	for t in cells:
		randDirs = np.random.normal(size=(len(newCells[t]),2))
		randDirs = 0.5*splitRad*randDirs/np.linalg.norm(randDirs, axis=1)[:, np.newaxis]
		cells[t] = np.concatenate((cells[t], cells[t][newCells[t]] + randDirs), axis=0)
		cells[t][newCells[t]] -= randDirs
	
	for t in cells:
		sc[t].set_offsets(cells[t])
	s=""
	for t in cells:
		s+=f"{t}: {len(cells[t])}, "

	global frames
	print(f"frame {frames}: {s[:-2]}")
	frames += 1
	return sc.values()

fig = pl.figure(figsize=(16,16))
sc={}
for t in cells:
	sc[t] = pl.scatter(cells[t][:,0],cells[t][:,1], s=2, color=colors[t])
pl.xlabel("x [μm]")
pl.ylabel("y [μm]")

anim = animation.FuncAnimation(fig, updatefig, interval=10, blit=True, frames=100_000)
writervideo = animation.FFMpegWriter(fps=30) 

outFolder = "out/tumorSim"
from pathlib import Path
Path(outFolder).mkdir(parents=True, exist_ok=True)
diff=str(list(switchMat.values()))
anim.save(f"{outFolder}/tumor_L={L}_diffusion={diff[1:-1]}_cancerGrowth={cancerGrowth}_immuneGrowth={immuneGrowth}.avi", writer=writervideo)

# pl.show()
