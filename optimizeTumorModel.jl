module Tst

include("TumorModel.jl")

import .TumorModelling

tm = TumorModelling.TumorModel([5,5,5], 0.17, [0,1,0], 30., [0 0 0])

L=5000.0 #
nCancer = 1_000
TumorModelling.simulate(tm, L, nCancer)

end