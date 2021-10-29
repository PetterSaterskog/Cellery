import pandas as pd
import numpy as np
import matplotlib.pyplot as pl

tumorCells = ['Tumor_cell', 'Tumor_Ki67+', 'Tumor']
immuneCells = ['Microglia', 'Macrophage', 'Neutrophil', 'T-cell']
brainCells = ['Neuron', 'PathCellObject', 'Astrocyte', 'Fibroblast']
vesselCells = ['Vessel_cell']

stromaCells = ['Macrophage', 'Vessel_cell', 'Fibroblast']

typeOrder = [*brainCells, *vesselCells, *tumorCells, *immuneCells]

cellMarkers = {'T-cell':('o', (0.2,1.0,0.2)),
			 'Macrophage':('^', (0.3,0.8,0.3)),
			 'PathCellObject':('^', (0.9,0.9,0.0)),
			 'Neuron':('^', 'royalblue'),
			 'Neutrophil':('^', 'forestgreen'),
			 'Microglia':('^', 'darkviolet'),
			 'Astrocyte':('^', 'crimson'),
			 'Tumor_Ki67+':('*', 'orange'),
			 'Tumor':('*', 'yellow'),
			 'Tumor_cell':('*','red'),
			 'Vessel_cell':('P','maroon'),
			 'Fibroblast':('P','gray'),
			 }

outFolder = f"out/exploreNewSpencerTumor"
from pathlib import Path
Path(outFolder).mkdir(parents=True, exist_ok=True)

def plotCells(cellsByType):
	tot = sum(len(cellsByType[t]) for t in cellsByType)
	# pl.gca().set_facecolor((0.5, 0.5, 0.5))
	for typeName in typeOrder:
		if not typeName in cellsByType: continue
		pCells = cellsByType[typeName]
		if len(pCells)==0: continue
		pl.plot(pCells[:,0],
				pCells[:,1],
				linestyle="None",
				marker=cellMarkers[typeName][0],
				color=cellMarkers[typeName][1],
				label = f"{typeName} ({100*pCells.shape[0]/tot:.1f}%)")
	pl.gca().axis('equal')
	pl.legend(loc="upper right")

def loadTumor(fileName):
	data = pd.read_csv(fileName)
	labelBy = 'Name' #'Class'
	return {c:data[[f'Centroid {c} µm' for c in 'XY']].to_numpy()[data[labelBy]==c] for c in pd.unique(data[labelBy])}

def getNormal(points):
	return np.array([points[0,1]-points[1,1], points[1,0]-points[0,0]]) / np.linalg.norm(points[1, :] - points[0, :])

def measureGradient(cellsByType, points, bins):
	n = getNormal(points)
	d = [-n[1], n[0]]
	center = points.mean(axis=0)
	w = np.linalg.norm(points[1, :] - points[0, :])
	depths = {k:[(c-center).dot(n) for c in v if abs((c-center).dot(d)) < w/2] for k,v in cellsByType.items()}
	return {k:np.histogram(v, bins)[0]/(np.diff(bins)*w) for k,v in depths.items()}, w

def findBorderDepth(bins, counts):
	inside = sum(counts[t] for t in tumorCells)
	outside = hist['Neuron']
	i = np.argmax(outside > inside)
	x = (outside[i-1]-inside[i-1]) / (outside[i-1] - inside[i-1] + inside[i] - outside[i])
	return ((bins[i-1]+bins[i])*(1-x) + (bins[i]+bins[i+1])*x)/2

def plotGradientMeasurement(points, bins, label=None):
	n = getNormal(points)
	pl.plot(*points.T, color='black', linestyle='--', lw=3)
	pl.plot(*(points+n[None, :]*bins[0]).T, color='black', linestyle='-', lw=3)
	pl.plot(*(points+n[None, :]*bins[-1]).T, color='black', linestyle='-', lw=3)
	if label: pl.text(*points[0], label, fontsize=40, color='black', weight='bold')