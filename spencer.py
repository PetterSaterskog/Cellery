from cell_distribution import CellDistribution
from collections import defaultdict
import numpy as np

inputDir = "images/spencer"

other = {"Image", "Name", "Detection probability"}

position = [
"Centroid X µm",
"Centroid Y µm"]

geometry={
"Cell: Area µm^2",
"Cell: Length µm",
"Cell: Circularity",
"Cell: Solidity",
"Distance to annotation Tumor µm",
"Distance to annotation Vessel µm",
"Distance to annotation Brain µm",
"Distance to annotation ECM µm",
"Delaunay: Num neighbors",
"Delaunay: Mean distance",
"Delaunay: Median distance",
"Delaunay: Max distance",
"Delaunay: Min distance"}

differentiators = {
"Cell: Area µm^2", #extra
"DAPI: Nucleus: Mean",
"CD3_Cy7_Rnd1: Cell: Mean",
"CD8a_Cy5_Rnd1: Cell: Mean",
"Iba1_Cy3_Rnd1: Cell: Mean",
"GFP_GFP_Rnd1: Cell: Mean",
"CD68_Cy7_Rnd2: Cell: Mean",
"P2Y12_Cy5_Rnd2: Cell: Mean",
"CD45_Cy3_Rnd2: Cell: Mean",
"HIF1a_GFP_Rnd2: Cell: Mean",
"Ly6b_Cy7_Rnd3: Cell: Mean",
"S100A8_Cy5_Rnd3: Cell: Mean",
"CC3_Cy3_Rnd3: Cell: Mean",
"NF-H_GFP_Rnd3: Cell: Mean",
"Olig2_Cy7_Rnd4: Nucleus: Mean",
"Ki67_Cy5_Rnd4: Nucleus: Mean",
"NeuN_Cy3_Rnd4: Nucleus: Mean",
"Lamin B1_GFP_Rnd4: Cell: Mean",
"Sox2_Cy7_Rnd5: Nucleus: Mean",
"VTN_Cy3_Rnd5: Cell: Mean",
"Lamin AC_GFP_Rnd5: Cell: Mean",
"CD13_Cy7_Rnd6: Cell: Mean",
"TNC_Cy5_Rnd6: Cell: Mean",
"Sox9_Cy3_Rnd6: Nucleus: Mean",
"aSMA_GFP_Rnd6: Cell: Mean",
"NG2_Cy7_Rnd7: Cell: Mean",
"Vimentin_GFP_Rnd7: Cell: Mean",
"MPO_Cy7_Rnd8: Cell: Mean",
"GFAP_Cy5_Rnd8: Cell: Mean",
"PDGFRb_Cy3_Rnd8: Cell: Mean",
"Osteopontin_Cy7_Rnd9: Cell: Mean",
"Periostin_Cy5_Rnd9: Cell: Mean",
"Desmin_GFP_Rnd9: Cell: Mean",
"CSPG5_Cy7_Rnd10: Cell: Mean",
"Fibronectin_Cy3_Rnd10: Cell: Mean",
"Phalloidin_GFP_Rnd10: Cell: Mean",
"Col I_Cy7_Rnd11: Cell: Mean",
"Col IV_Cy5_Rnd11: Cell: Mean",
"aTubulin_GFP_Rnd11: Cell: Mean",
"CD31_Cy5_Rnd12: Cell: Mean"}

def loadMarkers(fileName):
	ps = []
	ds = []
	with open(inputDir+'/'+fileName) as f:
		colNames = f.readline().split(',')
		markerColumns = {m: colNames.index(m) for m in differentiators}
		positionColumns = {m: colNames.index(m) for m in position}
		while True:# and len(ps)<1000:
			line = f.readline()
			if line:
				cols = line.split(',')
				ds.append([float(cols[markerColumns[m]]) for m in differentiators])
				ps.append([float(cols[positionColumns[m]]) for m in position])
			else: break
	return np.array(ps), np.array(ds)

def extractCellTypes(ds, nCellTypes = 15, minSize=100):
	import umap
	import hdbscan

	n_neighbors=30
	min_dist=.0
	n_components=5
	metric='euclidean'
	
	reducer = umap.UMAP(
		n_neighbors=n_neighbors,
		min_dist=min_dist,
		n_components=n_components,
		metric=metric
	)
	print(f"Zeroes: {ds[ds==0].flatten().shape[0]}")
	ds[ds==0] = 1
	data = np.log(ds)
	print("umap...")
	u = reducer.fit_transform(data)
	print("hdbscan...")
	labels = hdbscan.HDBSCAN(min_samples=nCellTypes, min_cluster_size=minSize).fit_predict(u)
	return labels

	# for i in range(0):
	# 	# data = np.log(cells[np.random.randint(len(cells), size=40_000)])
	# 	data = np.log(ds)
	# 	u = reducer.fit_transform(data)

	# 	labels = hdbscan.HDBSCAN(min_samples=10, min_cluster_size=200).fit_predict(u)
	# 	print(labels.shape)
	# 	fig = pl.figure()
	# 	clustered = labels >= 0
	# 	if n_components==3:
	# 		ax = fig.add_subplot(111, projection='3d')
	# 		ax.scatter(u[~clustered, 0], u[~clustered, 1], u[~clustered, 2], s=0.1, color=(0,0,0))
	# 		ax.scatter(u[clustered, 0], u[clustered, 1], u[clustered, 2], s=0.1, c=labels[clustered], cmap='Spectral')
	# 	if n_components==2:
	# 		pl.scatter(u[~clustered, 0], u[~clustered, 1], s=0.1, color=(0,0,0))
	# 		pl.scatter(u[clustered, 0], u[clustered, 1], s=0.1, c=labels[clustered], cmap='Spectral')
		
	# if False:
	# 	for i,m in enumerate(markers):
	# 		pl.figure()
	# 		pl.title(m)
	# 		pl.hist(np.log(cells[:, i]), bins = 100)
	# 		pl.gca().set_yscale("log")

	# pl.show()
	# exit(0)

	# stdCells = (cells - np.average(cells, axis=0)) / np.std(cells, axis=0)
	# stdCells = np.log(cells)
	# print(stdCells[:3])
	# cov = np.cov(np.array(stdCells).T)
	# vals, vecs = np.linalg.eigh(cov)
	# pcs = np.dot(stdCells, vecs)

	

	# pl.plot(pcs[:, 0], pcs[:, 1], linestyle='None', marker = '.')
	# pl.show()


if __name__ == "__main__":
	import matplotlib.pyplot as pl
	files = ["slide_1_measurements.csv", "slide_2_measurements.csv", "slide_3_measurements.csv"]
	expName = "8types"
	outFolder = f"out/spencer/{expName}"
	from pathlib import Path
	Path(outFolder).mkdir(parents=True, exist_ok=True)

	print("Loading...")
	cells = [loadMarkers(f) for f in files]
	markers = [c[1] for c in cells]
	allMarkers = np.concatenate([c[1] for c in cells])

	if True:
		types = extractCellTypes(allMarkers, nCellTypes = 8, minSize=150)
		n=0
		for i in range(len(markers)):
			typeFile = f"{inputDir}/{files[i][:-4]}types_{expName}.csv"
			np.savetxt(typeFile, types[n:n+len(markers[i])])
			n += len(markers[i])
	
	types = [np.loadtxt(f"{inputDir}/{f[:-4]}types_{expName}.csv") for f in files]
	
	markersByType = defaultdict(lambda: [])
	for i in range(len(files)):
		for j in range(len(markers[i])):
			markersByType[types[i][j]].append(markers[i][j])
	
	orderedMarkers = sorted(markersByType.keys())
	def markerName(m):
		return f"{m}: {100*markersByType[m].shape[0] / allMarkers.shape[0]:.1f}%"
	
	for m in orderedMarkers:
		markersByType[m] = np.array(markersByType[m])
		print(markerName(m))
	
	def plotHist(vals, linestyle, label, n=50, color=None, edges = None):
		if edges is None:
			edges = np.linspace(np.min(vals), np.max(vals), n+1)
		counts,_ = np.histogram(vals, edges)
		pl.plot(np.repeat(edges, 2)[1:-1], np.repeat(counts, 2), linestyle=linestyle, label=label, color=color)
		return edges
		
	def clipLog(a):
		return np.log(np.clip(a, 1, None))
	

	for i,d in enumerate(differentiators):
		pl.figure(figsize=(8, 6))
		pl.title(d)
		edges = plotHist(clipLog(allMarkers[:, i]), linestyle='-', label="all", color='black')
		for m in orderedMarkers:
			plotHist(clipLog(markersByType[m][:, i]), linestyle='-', label=markerName(m), edges=edges)
		pl.gca().set_yscale("log")
		pl.xlabel(f"log({d})")
		pl.legend()
		pl.savefig(f"{outFolder}/spencer_hists_{d}.pdf")
		pl.close()
	

	for i, f in enumerate(files):
		ps = cells[i][0]
		pl.figure(figsize=(50,50))
		for m in orderedMarkers:
			isForType = types[i]==m
			pl.plot(ps[isForType,0], ps[isForType,1], label = markerName(m), linestyle="None", marker='.')
		pl.legend(loc="upper right")
		pl.gca().axis('equal')
		pl.savefig(f"{outFolder}/spencer_celltypes_{f[:7]}.png")

		pl.figure(figsize=(14, 14))
		margin = 1700
		L= 5000
		for m in orderedMarkers:
			isForType = (types[i]==m) & (ps[:,0] > margin) & (ps[:,0] < L-margin) & (ps[:,1] > margin) & (ps[:,1] < L-margin)
			pl.plot(ps[isForType,0], ps[isForType,1], label = markerName(m), linestyle="None", marker='.')
		pl.gca().axis('equal')
		pl.legend(loc="upper right")
		pl.savefig(f"{outFolder}/spencer_celltypes_{f[:7]}_small.png")
	pl.show()	
	exit(0)

	if False:
		for f in files:
			typeFile = inputDir+"/"+f[:-4]+"_types.csv"

			if True:
				ps , ds = loadMarkers(f)
				types = extractCellTypes(ds, nCellTypes = 20)
				np.savetxt(typeFile, np.append(ps, np.array([types]).T, axis=1), delimiter=', ', header='x µm, y µm, type (-1 = unknown)')
			
			cells = np.loadtxt(typeFile, delimiter=', ', skiprows=1)
			pl.figure(figsize=(14, 14))
			pl.scatter(cells[:,0], cells[:,1], c=cells[:,2], s=0.1, cmap='hsv')
			pl.gca().axis('equal')
			pl.savefig(f"{outFolder}/spencer_celltypes_{f[:7]}.pdf")

			pl.figure(figsize=(14, 14))
			margin = 1700
			L= 5000
			inside =  (cells[:,0] > margin) & (cells[:,0] < L-margin) & (cells[:,1] > margin) & (cells[:,1] < L-margin)
			pl.scatter(cells[inside,0], cells[inside,1], c=cells[inside,2], s=3, cmap='hsv')
			pl.gca().axis('equal')
			pl.savefig(f"out/spencer/spencer_celltypes_{f[:7]}_small.pdf")
	pl.show()
	