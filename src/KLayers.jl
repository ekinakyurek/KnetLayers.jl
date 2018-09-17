module KLayers
using Knet
import Knet: save
export gpu,knetgc,KnetArray,relu,sigm,elu,invx,mat,
       Data,minibatch,train!,Train,param,param0,
       logp, logsumexp, nll, accuracy,zeroone,dropout,
       SGD, Sgd, Momentum, Nesterov, Adam, Adagrad, Adadelta, Rmsprop, update!, optimizers,
       gaussian, xavier, bilinear,setseed,
       hyperband, goldensection,RnnJLD,KnetJLD

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
Prm(x)=Param(atype(x));   export Prm
include("core.jl");       export Linear,Embed,Conv,Dense,BatchNorm
include("nonlinear.jl");  export ReLU,Sigm,Tanh,LeakyReLU,ELU,Dropout,LogP,SoftMax,LogSumExp
include("mlp.jl");        export MLP
include("rnn.jl");        export RNN,SRNN,LSTM,GRU

end # module
