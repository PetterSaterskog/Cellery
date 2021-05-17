from scipy.spatial import cKDTree

#doesn't sample pairs between a cell and itself if c2 is c1
def getCorr(c1, c2, edges, L, same):
	areas = np.diff(np.pi*edges**2)
	if same: #c2 is c1:
		pairs = cKDTree(c1).query_pairs(edges[-1], output_type =  'ndarray')
		if len(pairs):
			norm1 = 2 / areas / len(c1) / (len(c1)-1) * L**2
		else: norm1 = 1
	else:
		neighbors = cKDTree(c1).query_ball_tree(cKDTree(c2), edges[-1])
		pairs = np.array([(i, j) for i in range(len(c1)) for j in neighbors[i]])
		if len(pairs):
			norm1 = 1 / areas / len(c1) / len(c2) * L**2
		else: norm1 = 1
	if len(pairs)==0:
		dists = []
	else:
		dists = np.linalg.norm(c2[pairs[:,1]] - c1[pairs[:,0]], axis=1)
	hist = np.histogram(dists, edges)[0]
	return hist * norm1, np.sqrt(hist + 1) * norm1

def getTypeCorrelators(cs, edges, L):
	corrAndErr = {(ti,tj):getCorr(cs[ti], cs[tj], edges, L, ti==tj) for ti in range(len(cs)) for tj in range(len(cs)) if ti<=tj}
	return {k:v[0] for k,v in corrAndErr.items()}, {k:v[1] for k,v in corrAndErr.items()}
'''
def getTypeCorrelators3(cs, edges, L):
	ts = [2]
	# corrAndErr = {(ti,tj,tk):get3Corr((cs[ti], cs[tj], cs[tk]), edges, L, (ti,tj,tk)) for ti in range(len(cs)) for tj in range(len(cs)) for tk in range(len(cs)) if ti<=tj and tj<=tk}
	corrAndErr = {(ti,tj,tk):get3Corr((cs[ti], cs[tj], cs[tk]), edges, L, (ti,tj,tk)) for ti in ts for tj in ts for tk in ts if ti<=tj and tj<=tk}
	return {k:v[0] for k,v in corrAndErr.items()}, {k:v[1] for k,v in corrAndErr.items()}


def canonicalOrder(triplet, same):
	for i in range(len(triplet)):
		for j in range(len(triplet)):
			# if i<j and same[i] == same[j] and triplet[i] >= triplet[j]:
			if i!=j and same[i] == same[j] and triplet[i] == triplet[j]:
				return False
	return True

def get3Corr(cs, edges, L, same):
	areas = np.diff(np.pi*edges**2)

	trees = [cKDTree(c) for c in cs]
	neighbors01 = trees[0].query_ball_tree(trees[1], edges[-1])
	neighbors02 = trees[0].query_ball_tree(trees[2], edges[-1])

	triplets = [(i, j, k) for i in tqdm(range(len(cs[0]))) for j in neighbors01[i] for k in neighbors02[i]]
	canonicalTriplets = [t for t in tqdm(triplets) if canonicalOrder(t, same)]

	dists = np.linalg.norm([[cs[j][t[j]] - cs[(j+1)%3][t[(j+1)%3]] for j in range(3)] for t in tqdm(canonicalTriplets)], axis = 2)
	
	# This coordinate transformation maps the possible triangles (tri.-ineq.) _onto_ (R^+)^3
	coords = [dists[:, (i+1)%3] + dists[:, (i+2)%3] - dists[:, i] for i in range(3)]
	dets = np.sqrt(coords[0]*coords[1]*coords[2]*(coords[0]+coords[1]+coords[2]))/((coords[0]+coords[1])*(coords[1]+coords[2])*(coords[2]+coords[0]))
	hist = np.histogramdd(coords, [edges]*3, weights = dets)[0]
	return hist, np.sqrt(hist + 1)
	'''