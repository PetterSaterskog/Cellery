

import tumorModel
L=1000
tm = tumorModel.TumorModel(cellEffR={'cancer':5,'healthy':5,'immune':5},
immuneFraction=0.17,
growth={'cancer':1.0, 'immune':0}, diffusion = {('cancer', 'immune'):0.0, ('cancer', 'healthy'):0, ('healthy','immune'):0})

import time
s = time.time()
tumor = tumorModel.Tumor(tm, L=L, verbose=False, tumorCellCount = 3_000)
print(tumor.cellPositions.shape)
print(time.time()-s)