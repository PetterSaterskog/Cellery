import numpy as np
import matplotlib.pyplot as pl
import os
import pickle
from shapely.geometry import Polygon, Point

import tumorModel
import spencer

outFolder = f"out/necrosis"
from pathlib import Path
Path(outFolder).mkdir(parents=True, exist_ok=True)

#questions for spencer:
# thickness of slides, including segmentation
# expected behavior of tumor? Lots of diffusion? cytotoxicity? monoclonal?
# cell number densities?

# We assume each cell type has a fixed volume. We maximize the volumes under the constraint that no cell volumes overlap in any region of the image.
# Then we use these volumes to find the ratio of cells to empty space in different parts of the image. We attribute

thickness = 15

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
fig = pl.figure()
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

cellVolumes = np.linalg.solve(np.array([regionCounts[r] for r in regions]), [regions[r].area*thickness for r in regions])

cellEffR =  dict(zip(tumorModel.types, (cellVolumes/tumorModel.sphereVol(3))**(1/3)))
print(f"Cell effective radii by type: [h,c,i] = {cellEffR} μm")
tumorModel.plot(referencePositions, referenceTypes, cellEffR, fig=fig)
pl.savefig(f"{outFolder}/reference.png")


L=5000
edges = np.linspace(0, L, 500)
vols = np.histogramdd(referencePositions, bins = 2*[edges], weights = cellVolumes[referenceTypes])[0]

pl.figure()
pl.imshow(vols.T, origin='lower', extent = (0, L, 0, L))
pl.show()