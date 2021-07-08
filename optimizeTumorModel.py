# Petter Saterskog, 2021


# direct measurements:
# immuneFraction
# cell sizes
# tumor size
#
# remaining unknowns:
# immune growth
# cancer-healthy diffusion
# immune-healthy diffusion
#
# observables to fit
# number of immune cells in tumor, both in center and perifery
# island size in tumor, both in center and perifery
# decay length of tumor border into healthy tissue
# island shapes, are tumor or immune cells the islands? Are they jagged or round?
#
# We need to fit three indep parameters, immune cell "growth" (recruitment) speed, cancer-healthy diffusion, immune-healthy diffussion.
# We have set the immune-cancer diffusion to 0 by hand since the immune-cancer border is practically of 0 width in the center, and we assume all cells of a certain type obey the same rules at all times.
#
# This is doubly overdetermined with 3 parameters, and ~6 observables. A successful fit thus indicates that our model is an effective description of the tumor dynamics.

import numpy as np
import matplotlib.pyplot as pl
import os
from shapely.geometry import Polygon, Point

import tumorModel
import spencer

outFolder = f"out/optimizeTumorModel"
from pathlib import Path
Path(outFolder).mkdir(parents=True, exist_ok=True)

referencePositions =  spencer.loadMarkers("slide_1_measurements.csv")[0]
referenceTypes = np.loadtxt(f"{spencer.inputDir}/slide_1_measurementstypes_8types.csv")
refTypeIds = [0, 7, 3]
rowFilter = np.isin(referenceTypes, refTypeIds)
print(f"Kept {100*np.sum(rowFilter)/rowFilter.shape[0]:.1f}% of reference image cells")
referencePositions = referencePositions[rowFilter, :]
referenceTypes = np.array([refTypeIds.index(t) for t in referenceTypes[rowFilter]])
tumorModel.plot(referencePositions, referenceTypes)

#https://apps.automeris.io/wpd/
regionDir = f"{spencer.inputDir}/regions/slide1/"
regions = {}
for fn in os.listdir(regionDir):
	ext = np.loadtxt(regionDir + fn, delimiter = ",")
	pl.plot(ext[:,0], ext[:,1], label=fn[:-4], color=(1,0,1))
	regions[fn[:-4]] = Polygon(ext)
pl.savefig(f"{outFolder}/reference.png")
pl.close()

regions['healthy'] = regions['healthy1'].union(regions['healthy2'])
del regions['healthy1']
del regions['healthy2']
inRegion = {r:[regions[r].contains(Point(p)) for p in referencePositions] for r in regions}

regionCounts = {r:np.array([np.sum(referenceTypes[inRegion[r]]==ti) for ti in range(len(tumorModel.types))]) for r in regions}

# healthy oligodendrocytes are erroneously classified as cancer so we reclassify them since cancer is not expected outside the tumor
regionCounts['healthy'] = np.array([regionCounts['healthy'][0]+regionCounts['healthy'][1], 0, regionCounts['healthy'][2]])

for r in regions:
	print(f"Cell counts in {r} region: [h,c,i] = {100*regionCounts[r]/np.sum(regionCounts[r])}%")

immuneFraction = regionCounts['healthy'][2] / np.sum(regionCounts['healthy'])
print(f"Measured immune fraction in healthy tissue: {100*immuneFraction:.1f}%")

cellVolumes = np.linalg.solve(np.array([regionCounts[r] for r in regions]), [regions[r].area for r in regions])
cellEffR =  dict(zip(tumorModel.types, np.sqrt(cellVolumes/tumorModel.SphereVol(2))))
print(f"Cell effective radii by type: [h,c,i] = {cellEffR} μm")



# diffs = np.geomspace(0.5,3.0,4)
# imGrowths = np.geomspace(0.01,1.0,5)
L=500
for immuneGrowth in np.geomspace(0.01,0.1,3):
	for cancerDiff in np.geomspace(0.1,10,3):
		for immuneDiff in np.geomspace(0.25,25,3):
			tm = tumorModel.TumorModel(cellEffR=cellEffR, immuneFraction=immuneFraction, growth={'cancer':1.0, 'immune':immuneGrowth}, diffusion = {('cancer', 'immune'):0.0, ('cancer', 'healthy'):cancerDiff, ('healthy','immune'):immuneDiff})
			tumor = tumorModel.Tumor(tm, L=L, verbose=True, tumorCellRatio=0.95)
			tumorModel.plot(tumor.cellPositions, tumor.cellTypes)
			pl.savefig(f"{outFolder}/L={L}_immuneGrowth=_{immuneGrowth:.2f}_cancerDiff=_{cancerDiff:.2f}_immuneDiff=_{immuneDiff:.2f}.png")
			pl.close()