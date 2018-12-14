module KnetLayers

using Knet
import Knet: save, load
export gpu,knetgc,KnetArray,relu,sigm,elu,invx,mat,
       Data,minibatch,train!,Train,param,param0,params,
       logp, logsumexp, nll, bce, logistic, accuracy,zeroone,dropout,
       SGD, Sgd, Momentum, Nesterov, Adam, Adagrad, Adadelta, Rmsprop, update!, optimizers,
       gaussian, xavier, bilinear, setseed, train!,
       hyperband, goldensection, cpucopy, gpucopy,
       value, grad, cat1d, Param, @diff, @zerograd

"""
    KnetLayers.dir(path...)
Construct a path relative to KnetLayers root.
# Example
```julia
julia> KnetLayers.dir("src")
"/Users/ekin/git/KnetLayers/src"
```
"""
dir(path...) = joinpath(dirname(@__DIR__),path...)
seed! = Knet.seed!
#Array type for paramater initialization
arrtype = gpu()>=0 ? KnetArray{Float32} : Array{Float32}
#Setters for atype
settype!(t::T) where T<:Type{KnetArray{V}} where V <: AbstractFloat = gpu()>=0 ? (global arrtype=t) : error("No GPU available")
settype!(t::T) where T<:Type{Array{V}} where V <: AbstractFloat = (global arrtype=t)
settype!(t::Union{Type{KnetArray},Type{Array}}) = settype!(t{Float32})
#Param init function
Prm(x)=Param(atype(x));   export Prm


include("core.jl");
include("primitive.jl");   export Multiply, Embed, Linear, Dense, BatchNorm
include("nonlinear.jl");   export ReLU,Sigm,Tanh,LeakyReLU,ELU,Dropout,LogSoftMax,SoftMax,LogSumExp
include("loss.jl");        export CrossEntropyLoss, BCELoss, LogisticLoss
include("cnn.jl");         export Pool,UnPool,DeConv,Conv
include("special.jl");     export MLP
include("rnn.jl");         export RNN,SRNN,LSTM,GRU,RNNOutput,PadRNNOutput,PadSequenceArray
include("chain.jl");       export Chain


end # module
