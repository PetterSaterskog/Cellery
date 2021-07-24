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
import pickle
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
cellEffR =  dict(zip(tumorModel.types, np.sqrt(cellVolumes/tumorModel.sphereVol(2))))
print(f"Cell effective radii by type: [h,c,i] = {cellEffR} μm")


refNCellsByType = np.array([np.sum(referenceTypes == i) for i in range(len(tumorModel.types))])

L=6000
d=2
immuneGrowths = np.geomspace(0.005,0.03,3)
cancerDiffs = np.geomspace(3,30,2)
immuneDiffs = np.geomspace(0.05,1,2)
moveSpeeds = np.geomspace(0.1, 50.,4)
cases = [(L, d, cellEffR, immuneFraction, refNCellsByType, immuneGrowth, cancerDiff, immuneDiff, moveSpeed) for immuneGrowth in immuneGrowths for cancerDiff in cancerDiffs for immuneDiff in immuneDiffs for moveSpeed in moveSpeeds]

import subprocess


Path("tumorModels").mkdir(parents=True, exist_ok=True)
processes = []
for i in range(len(cases)):
	fn = f"tumorModels/{i}.pickle"
	pickle.dump( cases[i], open( fn, "wb" ) )
	processes.append(subprocess.Popen(["python3", "testTumorModel.py", fn]))

for p in processes:
	p.wait()
exit(0)

res = [[[np.loadtxt(f"{outFolder}/{getFileName(L, immuneGrowth, cancerDiff, immuneDiff)}_types.csv") for immuneGrowth in immuneGrowths] for cancerDiff in cancerDiffs] for immuneDiff in immuneDiffs]
nCellsByType = np.array([[[[np.sum(res[k][j][i] == l) for i in range(len(immuneGrowths))] for j in range(len(cancerDiffs))] for k in range(len(immuneDiffs))] for l in range(len(tumorModel.types))])

# for i in range(len(tumorModel.types)):
# 	pl.figure()
# 	pl.title(f"Number of {tumorModel.types[i]} cells")
# 	for j in range(len(immuneDiffs)):
# 		for k in range(len(cancerDiffs)):
# 			pl.plot(immuneGrowths, nCellsByType[i,j,k,:], label=f"immune cell diffusion = {immuneDiffs[j]:.2f}, cancer cell diffusion = {cancerDiffs[k]:.2f}")
# 	pl.plot(immuneGrowths, np.ones(immuneGrowths.shape)*refNCellsByType[i], label=f"Reference image", color = 'black', linestyle='--')
# 	pl.ylabel('count')
# 	pl.xlabel('immune cell growth speed')
# 	pl.legend(loc="upper right")

pl.figure()
pl.title(f"Ratio of immune to cancer cells")
for j in range(len(immuneDiffs)):
	for k in range(len(cancerDiffs)):
		pl.plot(immuneGrowths, nCellsByType[2,j,k,:]/nCellsByType[1,j,k,:], label=f"immune cell diffusion = {immuneDiffs[j]:.2f}, cancer cell diffusion = {cancerDiffs[k]:.2f}")
pl.plot(immuneGrowths, np.ones(immuneGrowths.shape)*refNCellsByType[2]/refNCellsByType[1], label=f"Reference image", color = 'black', linestyle='--')
pl.ylabel('ratio')
pl.xlabel('immune cell growth speed')
pl.legend(loc="upper right")

pl.show()
# resFiles = os.listdir(outFolder)

# for f in resFiles:
# 	if f.endswith("_types.csv"):
# 		np.savetxt(f"{outFolder}/{name}_types.csv", tumor.cellTypes)
