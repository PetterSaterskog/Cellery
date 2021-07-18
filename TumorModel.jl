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

	function update!(tm::TumorModel, dt::Real, cellPositions::Array{<:Real, 2}, cellTypes::Vector{CELL_TYPE}, L::Real)
		# tree = KDTree(transpose(data))
		toSplit = Vector{Int32}()
		for i = 1:size(cellTypes, 1)
			if cellTypes[i] == cancer && rand() < dt * tm.growth[Int(cancer)]
				append!( toSplit, i )
			end
		end

		r2 = tm.cellEffRadVec[1]^2
		displace = zeros(size(cellPositions))
		for i = 1:size(cellTypes, 1)
			for j = 1:size(toSplit, 1)
				diff =  cellPositions[i,1:2] - cellPositions[toSplit[j],1:2]
				n2 = sum(diff.*diff)
				if (n2 > 0) & (n2<100^2)
					displace[i,1:2] += diff * (sqrt(r2/n2 + 1) - 1)
				end
			end
		end
		cellPositions += displace
		
		randDirs = randn(size(toSplit,1), 2)
		randDirs ./= sqrt.(sum(randDirs.^2, dims=2))
		randDirs .*= tm.cellEffRadVec[Int(cancer)]
		cellPositions = [cellPositions; cellPositions[toSplit,:]+randDirs]
		cellPositions[toSplit, :] .-= randDirs
		cellTypes = cat(cellTypes, cellTypes[toSplit], dims=1)

		inside = (cellPositions[:,1] .> 0) .& (cellPositions[:,2] .> 0) .& (cellPositions[:,1] .< L) .& (cellPositions[:,2] .< L)
		cellPositions, cellTypes = cellPositions[inside,:], cellTypes[inside]

		return cellPositions, cellTypes
	end

	function createInitialCellDistribution(L::Real, d::Int, immuneRatio::Real, cellVolumes::Vector{<:Real})
		V = L^d
		averageCellVol = (1-immuneRatio)*cellVolumes[ Int(healthy)] + immuneRatio*cellVolumes[Int(immune)]
		n = V / averageCellVol
		nHealthy = trunc(Int, (1-immuneRatio)*n)
		nImmune = trunc(Int, immuneRatio*n)
		return L*[rand(nHealthy, d); [.5 .5]; rand(nImmune, d)], [fill(healthy, (nHealthy)); [cancer]; fill(immune, (nImmune))]
	end

	function simulate(tm:: TumorModel, L::Real, tumorCellCount::Int; verbose=false::Bool)
		dt = 0.2 / maximum([tm.growth; tm.diffusion]) #ensure the most common event only happens every 5 steps for small dt convergence

		cellPositions, cellTypes = createInitialCellDistribution(L, 2, 0.1, π*tm.cellEffRadVec.^2)
		nTumorCells = 1
		while nTumorCells < tumorCellCount
			cellPositions, cellTypes = update!(tm, dt, cellPositions, cellTypes, L)
			nTumorCells = count(x->x!=healthy, cellTypes)
			if verbose
				@Printf.printf("%d tumor cells\n", nTumorCells)
			end
		end
		return cellPositions, cellTypes
	end

	function plotTumor(cellPositions, cellTypes)
		return plot([layer(x=cellPositions[cellTypes.==t,1], y=cellPositions[cellTypes.==t,2], Geom.point, Theme(point_size=0.9mm, highlight_width=0mm, default_color=color(typeColors[t]))) for t in instances(CELL_TYPE)]...)
	end
end