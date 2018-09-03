module KLayers
import Knet: KnetArray,rnnforw,rnninit,xavier,BNMoments,Param,conv4,gpu,dropout
export KnetArray,rnnforw,rnninit,xavier,BNMoments,Param,conv4,gpu,dropout
import AutoGrad: @diff,@zerograd
export @diff,@zerograd

abstract type Model end

#Array type for paramater initialization
atype = gpu()>=0 ? KnetArray{Float32} : Array{Float32}

#Setters for atype
settype!(t::T) where T<:Type{KnetArray{V}} where V <: AbstractFloat = gpu()>=0 ? (global atype=t) : error("No GPU available")
settype!(t::T) where T<:Type{Array{V}} where V <: AbstractFloat = global atype=t
settype!(t::Union{Type{KnetArray},Type{Array}}) = settype!(t{Float32})

#Param init function
param(x)=Param(atype(x))

include("core.jl");      export Linear,Embed,Conv,BatchNorm
include("nonlinear.jl"); export ReLU,Sigm,Tanh,LeakyReLU,ELU,Dropout
include("mlp.jl");       export MLP
include("rnn.jl");       export RNN,SRNN,LSTM,GRU

end # module
