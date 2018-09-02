module KLayers

using Knet

atype = gpu()>=0 ? KnetArray{Float32} : Array{Float32}
settype!(t::T) where T<:Type{KnetArray{V}} where V <: AbstractFloat = gpu()>=0 ? (global atype=t) : error("No GPU available")
settype!(t::T) where T<:Type{Array{V}} where V <: AbstractFloat = global atype=t
settype!(t::Union{Type{KnetArray},Type{Array}}) = settype!(t{Float32})

param(x)=Param(atype(x))

include("core.jl"); export Linear,Embed,Conv,BacthNorm
include("mlp.jl");  export MLP
include("rnn.jl");  export RNN,SRNN,LSTM,GRU

end # module
