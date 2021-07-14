module Tst

	include("TumorModel.jl")

	import .TumorModelling

	tm = TumorModelling.TumorModel([5,5,5], 0.17, [0,1,0], 30., [0,0,0])

	L=5000.0 #
	nCancer = 1_000
	cellPositions, cellTypes = TumorModelling.simulate(tm, L, nCancer)
	using Gadfly
	import Cairo
	pl = TumorModelling.plotTumor(cellPositions, cellTypes)
	# display(pl)
	figPath = "out/sim/"
	mkpath(figPath)
	draw(PNG("out/sim/myplot.png", 30cm, 30cm), pl)
end