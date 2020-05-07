using KnetLayers, Test, Knet
gpu() < 0 ?  KnetLayers.settype!(Array{Float64}) : KnetLayers.settype!(KnetArray{Float64})
println("Testing with $(KnetLayers.arrtype)")
