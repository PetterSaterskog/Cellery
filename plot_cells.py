import numpy as np
import matplotlib.pyplot as pl
import matplotlib.patches as patches

figDir = "figs/"

TCellTypes = {'CD8-T','CD4-T','CD3-T','Tregs'}
BCellTypes = {'B'}
otherImmuneCellTypes = {'Macrophages', 'Other immune', 'Mono / Neu','DC / Mono','Neutrophils','DC','NK'}
cancerCellTypes = {'Kreatin-positive tumor','Tumor'}
otherCellTypes = {'Endothelial','Mesenchymal-like'}
unknownCellTypes = {'Unidentified'}
typeOrder = [*cancerCellTypes, *otherCellTypes, *TCellTypes, *BCellTypes, *otherImmuneCellTypes, *unknownCellTypes]

cellMarkers = {'CD8-T':('o', (0.9,0.4,0.6)),
			   'CD4-T':('o', (0.2,1.0,0.2)),
			   'CD3-T':('o', (0.4,0.4,0.9)),
			   'Tregs':('o', (0.9,0.4,0.9)),
			   'B':('$\mathrm{Y}$', (0.9,0.4,0.6)),
			   'Macrophages':('^', (0.3,0.8,0.3)),
			   'Other immune':('^', (0.5,0.5,1.0)),
			   'Mono / Neu':('^', (0.9,0.9,0.0)),
			   'DC / Mono':('^', 'royalblue'),
			   'Neutrophils':('^', 'forestgreen'),
			   'DC':('^', 'darkviolet'),
			   'NK':('^', 'crimson'),
			   'Kreatin-positive tumor':('*', 'orange'),
			   'Tumor':('*','red'),
			   'Endothelial':('P','maroon'),
			   'Mesenchymal-like':('P','gray'),
			   'Unidentified':('$?$','gray')
			   }

def plotCells(cellsByType, W, name, title):
	tot =  sum(len(cellsByType[t]) for t in cellsByType)
	pl.figure(figsize=(15, 15))
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

	pl.gca().add_patch(patches.Rectangle((0, 0), W, W,  linewidth=1, edgecolor='r', facecolor='none'))
	pl.legend(loc = 'right')
	pl.xlim((0, W*1.3))
	pl.ylim((0, W*1.3))
	pl.title(f'{title} - {tot} cells')
	pl.savefig(figDir+f"cellsByType_{name}.pdf")
	pl.close()

def plotHist(binEdges, counts, ls='-', label=''):
	pl.plot(np.repeat(binEdges, 2)[1:-1], np.repeat(counts, 2), ls, label=label)