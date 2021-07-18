module Tst

	include("TumorModel.jl")

	import .TumorModelling

	tm = TumorModelling.TumorModel([5,5,5] # radii
									, 0.17, [0,1,0], 30., [0,0,0])

	L=1000.0 #
	immuneFractio = 0.178
	cellEffectiveRadii = [10.54011730499363, 3.6393526483615, 3.875443759899738]
	nCancer = 3_000

	cellPositions, cellTypes = TumorModelling.simulate(tm, L, nCancer; verbose=true)
	print(size(cellPositions))
	using Gadfly
	import Cairo
	pl = TumorModelling.plotTumor(cellPositions, cellTypes)
	# display(pl)
	figPath = "out/sim/"
	mkpath(figPath)
	draw(PNG("out/sim/myplot.png", 30cm, 30cm), pl)
end