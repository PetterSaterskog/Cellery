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
# We have set the immune-cancer diffusion to 0 by hand since the immune-cancer border is practically of 0 s in the center, and we assume all cells of a certain type obey the same rules at all times.
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

# thickness = 15
# thickness = 25
thickness = 15
d=3

referencePositions =  spencer.loadMarkers("slide_1_measurements.csv")[0]
referenceTypes = np.loadtxt(f"{spencer.inputDir}/slide_1_measurementstypes_8types.csv")
refTypeIds = [0, 7, 3]
rowFilter = np.isin(referenceTypes, refTypeIds)
print(f"Kept {100*np.sum(rowFilter)/rowFilter.shape[0]:.1f}% of reference image cells")
referencePositions = referencePositions[rowFilter, :]
referenceTypes = np.array([refTypeIds.index(t) for t in referenceTypes[rowFilter]])

#https://apps.automeris.io/wpd/
regionDir = f"{spencer.inputDir}/regions/slide1/"
regions = {}
for fn in os.listdir(regionDir):
	ext = np.loadtxt(regionDir + fn, delimiter = ",")
	pl.plot(ext[:,0], ext[:,1], label=fn[:-4], color=(1,0,1))
	regions[fn[:-4]] = Polygon(ext)

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

cellVolumes = np.linalg.solve(np.array([regionCounts[r] for r in regions]), [regions[r].area*(1 if d==2 else thickness) for r in regions])

cellEffR =  dict(zip(tumorModel.types, (cellVolumes/tumorModel.sphereVol(d))**(1/d)))
print(f"Cell effective radii by type: [h,c,i] = {cellEffR} μm")
tumorModel.plot(referencePositions, referenceTypes, cellEffR)
pl.savefig(f"{outFolder}/reference.png")
pl.close()

refNCellsByType = np.array([np.sum(referenceTypes == i) for i in range(len(tumorModel.types))])

L = 3000
immuneGrowths = [0.0, 0.05, 0.1] #np.geomspace(0.005,0.02,3)[1:2]
cancerDiffs = [0.0, 1.0, 2.0] #np.geomspace(3,20,2)[:1]
immuneDiffs = [0.0, 1.0, 2.0] #np.geomspace(0.1,10,2)[:1]
moveSpeeds = [0.0, 1.0, 4.0]#np.geomspace(0.1, 50.,3)[1:2]

L = 5000
immuneGrowths = [0.025, 0.03, 0.04, 0.05] #np.geomspace(0.005,0.02,3)[1:2]
cancerDiffs = [3.0,5.0, 10.] #np.geomspace(3,20,2)[:1]
immuneDiffs = [0.0] #np.geomspace(0.1,10,2)[:1]
moveSpeeds = [0.0]#np.geomspace(0.1, 50.,3)[1:2]
cyctotoxicities = [0.3, 0.5, 0.7]

L = 4000
# thickness = 25
immuneGrowths = [0.035, 0.040, 0.045]
cancerDiffs = [2.5, 5.0] #np.geomspace(3,20,2)[:1]
immuneDiffs = [0.0] #np.geomspace(0.1,10,2)[:1]
moveSpeeds = [0.0]#np.geomspace(0.1, 50.,3)[1:2]
cyctotoxicities = [0.1, 0.3, 0.6, 1.0]

L = 4000
# thickness = 15
immuneGrowths = [0.026]
#neighborDist = 25
cancerDiffs = [5.0] #np.geomspace(3,20,2)[:1]
immuneDiffs = [0.0] #np.geomspace(0.1,10,2)[:1]
moveSpeeds = [0.0]#np.geomspace(0.1, 50.,3)[1:2]
cyctotoxicities = [0.1,0.2,0.5]

L = 4000
# thickness = 15
immuneGrowths = [0.15, 0.2, 0.25]
neighborDist = 15
cancerDiffs = [25., 50.0] #np.geomspace(3,20,2)[:1]
immuneDiffs = [0.0] #np.geomspace(0.1,10,2)[:1]
moveSpeeds = [0.0]#np.geomspace(0.1, 50.,3)[1:2]
cyctotoxicities = [1.0]

cases = [(L, d, cellEffR, immuneFraction, refNCellsByType, immuneGrowth, cancerDiff, immuneDiff, moveSpeed, cytotoxicity, thickness, neighborDist) for immuneGrowth in immuneGrowths for cancerDiff in cancerDiffs for immuneDiff in immuneDiffs for moveSpeed in moveSpeeds for cytotoxicity in cyctotoxicities]

np.random.shuffle(cases)

maxProcesses = 9

expName = '6'
exit(0)
import subprocess
Path("tumorModels").mkdir(parents=True, exist_ok=True)
processes = []
for i in range(len(cases)):
	# print('processes:', len(processes), processes)
	if len(processes)>=maxProcesses:
		processes[0].wait()
		processes.pop(0)
	
	print(f'Starting run {i}')
	fn = f"tumorModels/{expName}_{i}.pickle"
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
