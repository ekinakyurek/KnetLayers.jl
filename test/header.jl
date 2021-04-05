using KnetLayers, Test, Knet, CUDA
CUDA.functional() ?  KnetLayers.settype!(KnetArray{Float64}) : KnetLayers.settype!(Array{Float64})
println("Testing with $(KnetLayers.arrtype)")
