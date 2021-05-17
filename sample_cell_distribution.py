

#assumes equal sized bins starting at 0
def gContrib(ps, p, edges, norm2):
	# return np.histogram(np.linalg.norm(p[np.newaxis, :] - ps, axis = 1), bins = (len(edges)-1), range=(0, edges[-1]) )[0] * norm
	return np.histogram(np.linalg.norm(p[np.newaxis, :] - ps, axis = 1), bins = edges )[0] * norm2
	
def update(ps, ucorr, g, gErr, chi2, edges, L, ratios, norms):
	ti = np.random.choice(len(ps), p=ratios)
	i = np.random.randint(len(ps[ti]))
	newPos = np.random.rand(2) * L #np.remainder(ps[ti][i] + np.random.normal()*xSigma, L)

	newCorr = {}
	for oti in range(len(ps)):
		pair = (min(ti, oti), max(ti, oti))
		if oti==ti:
			newCorr[pair] = ucorr[pair] - gContrib(ps[ti][:i], ps[ti][i], edges, norms[pair])
										- gContrib(ps[ti][i+1:], ps[ti][i], edges, norms[pair])
										+ gContrib(ps[ti][:i], newPos, edges, norms[pair])
										+ gContrib(ps[ti][i+1:], newPos, edges, norms[pair])
		else:
			newCorr[pair] = ucorr[pair] - gContrib(ps[oti], ps[ti][i], edges, norms[pair])
										+ gContrib(ps[oti], newPos, edges, norms[pair])

	def getPartChi2(c1, c2):
		chi2 = 0
		for oti in range(len(ps)):
			pair = (min(ti, oti), max(ti, oti))
			chi2 += np.sum(((c1[pair] - c2[pair])/gErr[pair])**2)
		return chi2

	newChi2 = chi2 - getPartChi2(ucorr, g) + getPartChi2(newCorr, g)

	if newChi2 < chi2:
		ps[ti][i] = newPos
		chi2 = newChi2
		for oti in range(len(ps)):
			pair = (min(ti, oti), max(ti, oti))
			
			# ref = getTypeCorrelators(ps, edges, L)
			# print(newCorr[pair] - corr[pair][:])
			# print(ref[pair] - corr[pair][:])

			ucorr[pair][:] = newCorr[pair]
			# assert(False)
			# assert(np.linalg.norm(corr[pair] - ref[pair]) < 1e-6)
		return 1, chi2
	return 0, chi2

def getCellDistribution(nBatches=10, verbose=True):
	nBatches = 10
	
	for bi in range(nBatches):
		accepts = 0
		if verbose: print(f"\nBatch {bi+1}/{nBatches}")
		for j in range(nIts):
			accepted, chi2 = update(gPos, corr, g0, g0Err, chi2, sigma, edges, L, ratios, norms)
			accepts += accepted
		ar = accepts/nIts
		if verbose: print(f'Chi2: {chi2:.4g}')
		if verbose: print(f'Acceptance ratio: {ar}')
	return gPos