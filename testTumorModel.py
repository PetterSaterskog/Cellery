import sys
import pickle
import numpy as np
import matplotlib.pyplot as pl

import tumorModel


assert(len(sys.argv)==2)
L, d, cellEffR, immuneFraction, refNCellsByType, immuneGrowth, cancerDiff, immuneDiff, moveSpeed  = params = pickle.load( open( sys.argv[1], "rb" ) )

def getFileName(L, d, cellEffR, immuneFraction, refNCellsByType, immuneGrowth, cancerDiff, immuneDiff, moveSpeed):
	return f"L={L}_d={d}_immuneGrowth={immuneGrowth:.2f}_cancerDiff={cancerDiff:.2f}_immuneDiff={immuneDiff:.2f}_moveSpeed={moveSpeed:.2f}"

outFolder = f"out/optimizeTumorModel"
from pathlib import Path
Path(outFolder).mkdir(parents=True, exist_ok=True)

name = getFileName(*params)
tm = tumorModel.TumorModel(cellEffR=cellEffR, immuneFraction=immuneFraction, moveSpeed=moveSpeed, growth={'cancer':1.0, 'immune':immuneGrowth}, diffusion = {('cancer', 'immune'):0.0, ('cancer', 'healthy'):cancerDiff, ('healthy','immune'):immuneDiff})
tumor = tumorModel.Tumor(tm, L=L, d=d, verbose=False, tumorCellCount = (refNCellsByType[1]+refNCellsByType[2]))
np.savetxt(f"{outFolder}/{name}_positions.csv", tumor.cellPositions, delimiter = ', ')
np.savetxt(f"{outFolder}/{name}_types.csv", tumor.cellTypes)
tumorModel.plot(tumor.cellPositions, tumor.cellTypes)
pl.savefig(f"{outFolder}/{name}.png")
pl.close()

pl.figure(figsize=(10,10))
pl.imshow(tumor.infection[:,:], origin='lower')
pl.savefig(f"{outFolder}/{name}_infection.png")
pl.close()


print(f"{name} done!")
