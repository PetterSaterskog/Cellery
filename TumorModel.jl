module TumorModelling
export TumorModela, simulate
# export TumorModel

struct TumorModel
	cellEffRadVec
	immuneFraction::Real
	growth
	neighborDist::Real
	diffusion
end

function simulate(tm:: TumorModel, L::Real, nCancer:: Int)
	dt = 0.2 / max(np.max(list(tumorModel.growth.values())), np.max(tumorModel.diffusionMat)) #ensure the most common event only happens every 5 steps for small dt convergence
	println("hello world!")
end

end