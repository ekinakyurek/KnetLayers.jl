module KnetLayers

using Knet, AutoGrad, CUDA, LinearAlgebra
import Knet.KnetArrays: gc, save, load, _ser, JLDMODE
import Knet.Ops20: rnnforw, rnninit, RNN, BNMoments

export gpu, KnetArray,
       relu, sigm, elu, invx, mat, bmm,
       logp, logsumexp, nll, bce, logistic, accuracy, zeroone, dropout, softmax,
       Data, minibatch,
       param, param0, params,
       gaussian, xavier, bilinear, setseed,
       SGD, Momentum, Nesterov, Adam, Adagrad, Adadelta, Rmsprop, update!, optimizers,
       train!, converge!, adam!, sgd!, nesterov!, rmsprop!, progress!,
       adam, converge, train, sgd, nesterov, rmsprop, progress,
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
#Setters for atype
arrtype = Array{Float32}

"""
    Used for setting default underlying array type for layer parameters.

    settype!(t::T) where T<:Type{KnetArray{V}} where V <: AbstractFloat = CUDA.functional() ? (global arrtype=t) : error("No GPU available")
    settype!(t::T) where T<:Type{Array{V}} where V <: AbstractFloat = (global arrtype=t)
    settype!(t::Union{Type{KnetArray},Type{Array}}) = settype!(t{Float32})

# Example
```julia
julia> KnetLayers.settype!(KnetArray) # on a GPU machine
KnetArray{Float32}
```
"""
settype!(t::T) where T<:Type{KnetArray{V}} where V <: AbstractFloat = CUDA.functional() ? (global arrtype=t) : error("No GPU available")
settype!(t::T) where T<:Type{Array{V}} where V <: AbstractFloat = (global arrtype=t)
settype!(t::Union{Type{KnetArray},Type{Array}}) = settype!(t{Float32})

include("core.jl");
include("primitive.jl");   export Bias, Multiply, Embed, Linear, Dense, BatchNorm, Diagonal, LayerNorm
include("nonlinear.jl");   export NonAct, ReLU,Sigm,Tanh,LeakyReLU,ELU, Dropout, LogSoftMax, SoftMax, LogSumExp, GeLU
include("loss.jl");        export CrossEntropyLoss, BCELoss, LogisticLoss, SigmoidCrossEntropyLoss
include("cnn.jl");         export Pool,UnPool,DeConv,Conv
include("special.jl");     export MLP
include("rnn.jl");         export RNN,SRNN,LSTM,GRU,RNNOutput,PadRNNOutput,PadSequenceArray
include("chain.jl");       export Chain
include("attention.jl");   export MultiheadAttention
include("transformer.jl"); export Transformer, TransformerDecoder, PositionEmbedding, TransformerModel
include("datasets/Datasets.jl"); export Datasets
include("../data/IndexedDict.jl");
include("../data/one_hot.jl");


function __init__()
    global arrtype = CUDA.functional() ? KnetArray{Float32} : Array{Float32}
end

end # module
