module TumorModelling
	export TumorModela, simulate, plotTumor

	import Printf
	using Random
	using NearestNeighbors
	using Gadfly

	@enum CELL_TYPE healthy=1 cancer=2 immune=3
	typeColors = Dict([(healthy, "green"), (cancer, "red"), (immune, "blue")])

	struct TumorModel
		cellEffRadVec
		immuneFraction::Real
		growth
		neighborDist::Real
		diffusion
	end

	function update!(tm::TumorModel, dt::Real, cellPositions::Array{<:Real, 2}, cellTypes::Vector{CELL_TYPE})
		# tree = cKDTree(self.cellPositions)
		# pairs = tree.query_pairs(tumorModel.neighborDist, output_type="ndarray")
		
		# newCells = []
		# displacements = np.zeros(self.cellPositions.shape)

		# for i in range(len(types)):
		# 	for j in range(len(types)):
		# 		if tumorModel.diffusionMat[i,j]>0:
		# 			allPairs = pairs[(self.cellTypes[pairs[:,0]]==i) & (self.cellTypes[pairs[:,1]]==j), :]
		# 			for p in np.random.permutation(allPairs[np.random.rand(allPairs.shape[0]) < dt * tumorModel.diffusionMat[i,j]]): #permute to avoid finite dt effects being biased based on nn search order
		# 				if self.cellTypes[p[0]]==i and self.cellTypes[p[1]]==j: #don't swap cells that have already been swapped..
		# 					self.cellTypes[p[0]], self.cellTypes[p[1]] = self.cellTypes[p[1]], self.cellTypes[p[0]]

		# ccs = np.arange(len(self.cellPositions))[self.cellTypes==1]
		# ics = np.concatenate( (pairs[(self.cellTypes[pairs[:,0]]==2) & (self.cellTypes[pairs[:,1]]==1), 0],
		# 						pairs[(self.cellTypes[pairs[:,1]]==2) & (self.cellTypes[pairs[:,0]]==1), 1]) )

		# toSplit = np.concatenate( (ccs[np.random.rand(ccs.shape[0]) < dt*tumorModel.growth['cancer']], ics[np.random.rand(ics.shape[0]) < dt*tumorModel.growth['immune']] ))

		# diffs = self.cellPositions[:, np.newaxis,:] - self.cellPositions[np.newaxis, toSplit,:]
		# ns = np.sum(diffs**2, axis=2)
		# ns[ns==0] = 1
		# self.cellPositions += np.sum(diffs * (np.sqrt((tumorModel.cellEffRadVec[self.cellTypes[toSplit]]**2)[np.newaxis,:]/ns + 1) - 1)[:,:,np.newaxis], axis=1)

		# randDirs = np.random.normal(size=(len(toSplit),2))
		# randDirs = tumorModel.cellEffRadVec[self.cellTypes[toSplit]][:, np.newaxis]*randDirs/np.linalg.norm(randDirs, axis=1)[:, np.newaxis]
		# self.cellPositions = np.concatenate((self.cellPositions, self.cellPositions[toSplit] + randDirs), axis=0)
		# self.cellPositions[toSplit] -= randDirs
		# self.cellTypes = np.concatenate((self.cellTypes, self.cellTypes[toSplit]), axis=0)

		# inside = np.all((self.cellPositions > 0) & (self.cellPositions < self.L), axis=1)
		# self.cellPositions, self.cellTypes = self.cellPositions[inside], self.cellTypes[inside]
		# return toSplit.shape[0]
		return 1
	end

	function createInitialCellDistribution(L::Real, d::Int, immuneRatio::Real, cellVolumes::Vector{<:Real})
		nHealthy = 10_000
		nImmune = 100
		return L*[rand(nHealthy, d); [.5 .5]; rand(nImmune, d)], [fill(healthy, (nHealthy)); [cancer]; fill(immune, (nImmune))]
	end

	function simulate(tm:: TumorModel, L::Real, tumorCellCount::Int, verbose=false::Bool)

		dt = 0.2 / maximum([tm.growth; tm.diffusion]) #ensure the most common event only happens every 5 steps for small dt convergence

		cellPositions, cellTypes = createInitialCellDistribution(L, 2, 0.1, π*tm.cellEffRadVec.^2)
		nTumorCells = 1
		while nTumorCells < tumorCellCount
			nTumorCells += update!(tm, dt, cellPositions, cellTypes)
			if verbose
				@Printf.printf("%d tumor cells\n", nTumorCells)
			end
		end
		# println([tm.growth; tm.diffusion])
		# @Printf.printf("dt = %.2f s \n", dt)
		return cellPositions, cellTypes
	end

	function plotTumor(cellPositions, cellTypes)
		return plot([layer(x=cellPositions[cellTypes.==t,1], y=cellPositions[cellTypes.==t,2], Geom.point, Theme(point_size=0.3mm, highlight_width=0mm, default_color=color(typeColors[t]))) for t in instances(CELL_TYPE)]...)
	end
end