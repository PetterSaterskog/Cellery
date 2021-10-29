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

import os
import numpy as np
import matplotlib.pyplot as pl
from shapely.geometry import Polygon, Point

import tumor_model as tm

import tumorModel
import spencer
from spencerNewUtils import *
from cycler import cycler

outFolder = f"out/optimizeTumorModelRust"
from pathlib import Path
Path(outFolder).mkdir(parents=True, exist_ok=True)

thickness = 15
d=3

L = 1000
# thickness = 15
immuneGrowths = [0, 0.026]
neighborDist = 25
cancerDiffs = [0, 5.0] #np.geomspace(3,20,2)[:1]
immuneDiffs = [0.0] #np.geomspace(0.1,10,2)[:1]
moveSpeeds = [0.0]#np.geomspace(0.1, 50.,3)[1:2]
cyctotoxicities = [0.0, 0.2, 0.5, 1.0]

cellsByType = loadTumor(f"images/spencer/new.csv")

referencePositions = []
referenceTypes = []
cats = [['Neuron'], tumorCells, stromaCells]
for t in cellsByType:
	for i, cat in enumerate(cats):
		if t in cat:
			for c in cellsByType[t]:
				referencePositions.append(c)
				referenceTypes.append(i)
referencePositions = np.array(referencePositions)
referenceTypes = np.array(referenceTypes)
# referencePositions = spencer.loadMarkers("slide_1_measurements.csv")[0]
# referenceTypes = np.loadtxt(f"{spencer.inputDir}/slide_1_measurementstypes_8types.csv")
# refTypeIds = [0, 7, 3]
# rowFilter = np.isin(referenceTypes, refTypeIds)
# print(f"Kept {100*np.sum(rowFilter)/rowFilter.shape[0]:.1f}% of reference image cells")
# referencePositions = referencePositions[rowFilter, :]
# referenceTypes = np.array([refTypeIds.index(t) for t in referenceTypes[rowFilter]])

#https://apps.automeris.io/wpd/
regionDir = f"{spencer.inputDir}/regions/slide1/"
regions = {}
pl.figure(figsize=(14,14))
custom_cycler = (cycler(color=['c', 'm', 'y', 'k']) + cycler(lw=[1, 2, 3, 4]))
pl.gca().set_prop_cycle(custom_cycler)
for fn in os.listdir(regionDir):
	ext = np.loadtxt(regionDir + fn, delimiter = ",")
	ext += [700,500]
	extLoop = np.concatenate([ext[-1:,:], ext], axis=0)
	pl.plot(extLoop[:,0], extLoop[:,1], label=fn[:-4], linewidth=4)#, color=(1,0,1))
	regions[fn[:-4]] = Polygon(ext)

regions['healthy'] = regions['healthy1'].union(regions['healthy2'])
del regions['healthy1']
del regions['healthy2']

inRegion = {r:[regions[r].contains(Point(p)) for p in referencePositions] for r in regions}
print(regions.keys())
print(inRegion.keys())
regionCounts = {r:np.array([np.sum(referenceTypes[inRegion[r]]==ti) for ti in range(len(tumorModel.types))]) for r in regions}

# healthy oligodendrocytes are erroneously classified as cancer so we reclassify them since cancer is not expected outside the tumor
regionCounts['healthy'] = np.array([regionCounts['healthy'][0]+regionCounts['healthy'][1], 0, regionCounts['healthy'][2]])

for r in regions:
	print(f"Cell counts in {r} region: [h,c,i] = {100*regionCounts[r]/np.sum(regionCounts[r])}%")

immuneFraction = regionCounts['healthy'][2] / np.sum(regionCounts['healthy'])
print(f"Measured immune fraction in healthy tissue: {100*immuneFraction:.1f}%")

cellVolumes = np.linalg.solve(np.array([regionCounts[r] for r in regions]), [regions[r].area*(1 if d==2 else thickness) for r in regions])
cellEffR = dict(zip(tumorModel.types, (cellVolumes/tumorModel.sphereVol(d))**(1/d)))
print(f"Cell effective radii by type: [h,c,i] = {cellEffR} μm")

if True:
	print("Plotting reference image..")
	tumorModel.plot(referencePositions, referenceTypes, cellEffR, fig=pl.gcf())
	pl.savefig(f"{outFolder}/reference.png")
	pl.close()
	exit(0)

print("Solving for cell sizes..")
refNCellsByType = np.array([np.sum(referenceTypes == i) for i in range(len(tumorModel.types))])

cases = [(L, d, cellEffR, immuneFraction, refNCellsByType, immuneGrowth, cancerDiff, immuneDiff, moveSpeed, cytotoxicity, thickness, neighborDist) for immuneGrowth in immuneGrowths for cancerDiff in cancerDiffs for immuneDiff in immuneDiffs for moveSpeed in moveSpeeds for cytotoxicity in cyctotoxicities]

def getFileName(L, d, cellEffR, immuneFraction, refNCellsByType, immuneGrowth, cancerDiff, immuneDiff, moveSpeed, cytotoxicity, thickness, neighborDist):
	return f"L={L}_d={d}_immuneGrowth={immuneGrowth}_cancerDiff={cancerDiff}_immuneDiff={immuneDiff:.2f}_moveSpeed={moveSpeed:.2f}_cytotoxicity={cytotoxicity:.2f}_thicknesss={thickness:.2f}_neighborDist={neighborDist:.1f}"

print("Simulating tumor growths..")
for c in cases[:1]:
	L, d, cellEffR, immuneFraction, refNCellsByType, immuneGrowth, cancerDiff, immuneDiff, moveSpeed, cytotoxicity, thickness, neighborDist = c
	model = tm.TumorModel([cellEffR['healthy'], cellEffR['cancer'], cellEffR['immune']])
	
	nCells = L**3 / ((1-immuneFraction)*cellEffR['healthy']**d + immuneFraction*cellEffR['immune']**d) / tumorModel.sphereVol(d)
	nHealthy = int(nCells*(1-immuneFraction))
	nImmune = int(nCells*immuneFraction)
	cellPositions, cellTypes = tm.run(model, L, nHealthy, nImmune, 100_000, thickness)
	# print(cellPositions)
	# print(cellTypes)

	pl.figure(figsize=(14, 14))
	tumorModel.plot(cellPositions, cellTypes, cellEffR, L=L, fig=pl.gcf())
	pl.savefig(f"{outFolder}/{getFileName(*c)}.png")
	pl.close()