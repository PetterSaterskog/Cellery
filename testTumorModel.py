import sys
import pickle
import numpy as np
import matplotlib.pyplot as pl

import tumorModel

assert(len(sys.argv)==2)
L, d, cellEffR, immuneFraction, refNCellsByType, immuneGrowth, cancerDiff, immuneDiff, moveSpeed, cytotoxicity, thickness, neighborDist  = params = pickle.load( open( sys.argv[1], "rb" ) )

def getFileName(L, d, cellEffR, immuneFraction, refNCellsByType, immuneGrowth, cancerDiff, immuneDiff, moveSpeed, cytotoxicity, thickness, neighborDist):
	return f"L={L}_d={d}_immuneGrowth={immuneGrowth}_cancerDiff={cancerDiff}_immuneDiff={immuneDiff:.2f}_moveSpeed={moveSpeed:.2f}_cytotoxicity={cytotoxicity:.2f}_thicknesss={thickness:.2f}_neighborDist={neighborDist:.1f}"

outFolder = f"out/optimizeTumorModel"
from pathlib import Path
Path(outFolder).mkdir(parents=True, exist_ok=True)

if d==2:
	tumorSize = (refNCellsByType[1]+refNCellsByType[2])*0.5
if d==3:
	tumorSize = (refNCellsByType[1]+refNCellsByType[2])*L/thickness*0.7#*0.05

name = getFileName(*params)
print(f"Testing: {name}")

tm = tumorModel.TumorModel(cellEffR=cellEffR, immuneFraction=immuneFraction, moveSpeed=moveSpeed, cytotoxicity=cytotoxicity, neighborDist=neighborDist, growth={'cancer':1.0, 'immune':immuneGrowth}, diffusion = {('cancer', 'immune'):0.0, ('cancer', 'healthy'):cancerDiff, ('healthy','immune'):immuneDiff})
tumor = tumorModel.Tumor(tm, L=L, d=d, tumorCellCount = tumorSize, maxSteps = 10000, verbose=True, width=thickness, saveEvolution=True)
np.savetxt(f"{outFolder}/{name}_positions.csv", tumor.cellPositions, delimiter = ', ')
np.savetxt(f"{outFolder}/{name}_types.csv", tumor.cellTypes)
fig, sc = tumorModel.plot(tumor.cellPositions, tumor.cellTypes, tm.cellEffR, L=L, width=thickness)
pl.savefig(f"{outFolder}/{name}.png")

frameSkip = 1
def updateFig(frameI):
	evI = frameI*frameSkip
	for i in range(len(tumorModel.types)):
		circles = [pl.Circle((xi,yi), radius=tm.cellEffR[tumorModel.types[i]]*0.3, linewidth=0, label=tumorModel.types[i]) for xi,yi in tumor.evolution[evI][0][tumor.evolution[evI][1]==i]]
		sc[i].set_paths(circles)
		# sc[i].set_offsets(tumor.evolution[evI][0][tumor.evolution[evI][1]==i])
	return sc

import matplotlib.animation as animation
anim = animation.FuncAnimation(fig, updateFig, frames=len(tumor.evolution)//frameSkip, blit=True)
writervideo = animation.FFMpegWriter(fps=30) 
anim.save(f"{outFolder}/{name}.avi", writer=writervideo)

# if d==2:
# 	pl.figure(figsize=(10,10))
# 	pl.imshow(tumor.infection.T, origin='lower')
# 	pl.savefig(f"{outFolder}/{name}_infection.png")
# 	pl.close()

# 	pl.figure(figsize=(10,10))
# 	pl.imshow(tumor.exclusion.T, origin='lower')
# 	pl.savefig(f"{outFolder}/{name}_exclusion.png")
# 	pl.close()

# 	pl.figure(figsize=(10,10))
# 	pl.imshow(tumor.exclusion.T<0.5, origin='lower')
# 	pl.savefig(f"{outFolder}/{name}_exclusion_bin.png")
# 	pl.close()

print(f"{name} done!")
