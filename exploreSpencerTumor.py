from spencerNewUtils import *

cellsByType = loadTumor(f"images/spencer/new.csv")

borderSegments = np.array([[[1410, 3715], [1620, 4110]], [[1426, 2125], [1231, 3321]], [[2854., 1163.], [2190., 1318.]], [[3742, 1457], [3379, 1243]], [[4568, 2098], [4248, 1768]]])
bins = np.linspace(-750, 500, 19)
hists = []
ws = []
shiftedPoints = []
for i, points in enumerate(borderSegments):
	shift = 0
	print('Optimizing bin placement..')
	for _ in range(7):
		hist, _ = measureGradient(cellsByType, points + shift*getNormal(points)[None, :], bins)
		shift += findBorderDepth(bins, hist)/2
		print(shift)
	shiftedPoints.append(points + shift*getNormal(points)[None, :])
	hist, w = measureGradient(cellsByType, shiftedPoints[i], bins)
	hists.append(hist)
	ws.append(w)

plotCats = [(tumorCells, 'Tumor cells'), (['Neuron'], 'Neurons'), (['Microglia'], 'Microglia'), (['Macrophage'], 'Macrophages')] #, (stromaCells, 'Stroma')]
binCenters = (bins[:-1] + bins[1:])/2

fig, axs = pl.subplots(2, 2, figsize=(8,6))
for i, (types, label) in enumerate(plotCats):
	pl.sca(axs[i//2, i%2])
	pl.title(label)
	for j, hist in enumerate(hists):
		pl.plot(binCenters, 1e3*sum(hist[t] for t in types), label=chr(ord("A")+j))
	pl.grid()
	pl.legend()
	pl.xlabel('Outward distance from tumor border [μm]')
	pl.ylabel('Cell density [mcells/μm²]')
pl.tight_layout()
pl.savefig(f"{outFolder}/siteVariation.pdf")

def plotCategoryHist(cats, label=None):
	ys = [sum(h[t] for t in cats) for h in hists]
	meanY = np.average(ys, axis=0, weights=ws)
	n = len(hists)
	stdMeanY = np.sqrt(np.average((ys - meanY)**2, axis=0, weights=ws)/(len(hists)-1))
	# +f", $\\rho_{{max}}$ = {np.max(meanY)*1e3:.1f} mcells/μm²"
	pl.errorbar(binCenters, meanY/np.max(meanY), stdMeanY/np.max(meanY), label=label, capsize=3)

pl.figure(figsize=(5, 4))
for (types, label) in plotCats:
	plotCategoryHist(types, label=label)
# plotCategoryHist(['Vessel_cell'], label='Vessel_cell')
pl.legend(loc='upper right', ncol=2)
pl.xlabel('Outward distance from tumor border [μm]')
pl.ylabel('Normalized density of cells')
pl.grid()
pl.savefig(f"{outFolder}/siteAverage.pdf")

pl.figure(figsize=(25,25))
plotCells(cellsByType)
for i, points in enumerate(shiftedPoints):
	plotGradientMeasurement(points, bins, label=chr(ord("A")+i))
pl.savefig(f"{outFolder}/tumor.png")

pl.show()