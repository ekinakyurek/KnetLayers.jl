"""

    SRNN(inputSize,hiddenSize;activation=:relu,options...)
    LSTM(inputSize,hiddenSize;options...)
    GRU(inputSize,hiddenSize;options...)

    (1) (l::T)(x;kwargs...) where T<:RNN
    (2) (l::T)(x::Array{Int};batchSizes=nothing,kwargs...) where T<:RNN
    (3) (l::T)(x::Vector{Vector{Int}};sorted=false,kwargs...) where T<:RNN

All RNN layers has above forward run(1,2,3) functionalities.

(1) `x` is an input array with size equals d,[B,T]

(2) For this You need to have an `RNN` with embedding layer.
`x` is an integer array and inputs coressponds one hot vector indices.
You can give 2D array for minibatching as rows corresponds to one instance.
You can give 1D array with minibatching by specifying batch batchSizes argument.
Checkout `Knet.rnnforw` for this.

(3) For this You need to have an `RNN` with embedding layer.
`x` is an vector of integer vectors. Every integer vector corresponds to an
instance. It automatically batches inputs. It is better to give inputs as sorted.
If your inputs sorted you can make `sorted` argument true to increase performance.

Outputs of the forward functions are always `y,h,c,indices`.
`h`,`c` and `indices` may be nothing depending on the kwargs you used in forward.

`y` is last hidden states of each layer. `size(y)=(H/2H,[B,T])`.
If you use batchSizes argument `y` becomes 2D array `size(y)=(H/2H,sum(batchSizes))`.
To get correct hidden states for an instance in your batch you should use
indices output.

`h` is the hidden states in each timesstep. `size(h) = h,B,L`

`c` is the hidden states in each timesstep. `size(h) = h,B,L/2L`

`indices` is corresponding indices for your batches in `y` if you used batchSizes.
To get ith instance's hidden states in each times step,
you may type: `y[:,indices[i]]`
`

# options

* `embed=nothing`: embedding size or and embedding layer
* `numLayers=1`: Number of RNN layers.
* `bidirectional=false`: Create a bidirectional RNN if `true`.
* `dropout=0`: Dropout probability. Ignored if `numLayers==1`.
* `skipInput=false`: Do not multiply the input with a matrix if `true`.
* `dataType=Float32`: Data type to use for weights.
* `algo=0`: Algorithm to use, see CUDNN docs for details.
* `seed=0`: Random number seed for dropout. Uses `time()` if 0.
* `winit=xavier`: Weight initialization method for matrices.
* `binit=zeros`: Weight initialization method for bias vectors.
* `usegpu=(gpu()>=0)`: GPU used by default if one exists.

# kwargs
* hx=nothing : initial hidden states
* cx=nothing : initial memory cells
* hy=false   : if true returns h
* cy=false   : if true return c

"""
abstract type RNN <: Model end

struct SRNN <: RNN
    embedding::Union{Nothing,Embed}
    params
    specs
    gatesview::Dict
end
function SRNN(input::Int,hidden::Int;embed=nothing,activation=:relu,o...)
    embedding,inputSize = _getEmbed(input,embed)
    r,w = rnninit(inputSize,hidden;rnnType=activation,o...)
    gatesview  = Dict()
    p = param(w)
    for l=1:r.numLayers, (ih,ihid) in ihmaps, (ty,param) in wbmaps
        gatesview["$(ty)_$(ih)_l$(l)"] = rnnparam(r,p,l,ihid,param;useview=true)
    end
    SRNN(embedding,p,r,gatesview)
end
(m::SRNN)(x,h...;o...) = _forw(m,x,h...;o...)

const lstmmaps = Dict(:i=>(1,5),:f=>(2,6),:n=>(3,7),:o=>(4,8))
const ihmaps   = Dict(:i=>1,:h=>2)
const wbmaps   = Dict(:w=>1,:b=>2)

struct LSTM <: RNN
    embedding::Union{Nothing,Embed}
    params
    specs
    gatesview::Dict
end
function LSTM(input::Int,hidden::Int;embed=nothing,o...)
    embedding,inputSize = _getEmbed(input,embed)
    r,w = rnninit(inputSize,hidden;rnnType=:lstm,o...)
    gatesview  = Dict()
    p = param(w)
    for l=1:r.numLayers,(g,id) in lstmmaps,(ih,ihid) in ihmaps,(ty,param) in wbmaps
        gatesview["$(ty)_$(ih)_$(g)_l$(l)"] = rnnparam(r,p,l,id[ihid],param;useview=true)
    end
    LSTM(embedding,p,r,gatesview)
end
(m::LSTM)(x,h...;o...) = _forw(m,x,h...;o...)

const grumaps  = Dict(:r=>(1,4),:u=>(2,5),:n=>(3,6))
struct GRU <: RNN
    embedding::Union{Nothing,Embed}
    params
    specs
    gatesview::Dict
end

function GRU(input::Int,hidden::Int;embed=nothing,o...)
    embedding,inputSize = _getEmbed(input,embed)
    r,w = rnninit(inputSize,hidden;rnnType=:gru,o...)
    gatesview  = Dict()
    p = param(w)
    for l=1:r.numLayers, (g,id) in grumaps, (ih,ihid) in ihmaps,(ty,param) in wbmaps
        gatesview["$(ty)_$(ih)_$(g)_l$(l)"] = rnnparam(r,p,l,id[ihid],param;useview=true)
    end
    GRU(embedding,p,r,gatesview)
end
(m::GRU)(x,h...;o...) = _forw(m,x,h...;o...)

_getEmbed(input::Int,embed::Nothing)   = (nothing,input)
_getEmbed(input::Int,embed::Embed)     = size(embed.w,2) == input ? (embed,input) : error("dimension mismatch in your embedding")
_getEmbed(input::Int,embed::Integer)   = (Embed(input,embed),embed)

function _forw(rnn::RNN,seq::Array{T},h...;batchSizes=nothing,o...) where T<:Integer
    rnn.embedding === nothing && error("rnn has no embedding!")
    ndims(seq) == 1 && batchSizes === nothing && (seq = reshape(seq,1,length(seq)))
    y,h,c,_ = rnnforw(rnn.specs,rnn.params,rnn.embedding(seq),h...;batchSizes=batchSizes,o...)
    return y,h,c,nothing
end

function _forw(rnn::RNN,batch::Vector{Vector{T}},h...;sorted=true,o...) where T<:Integer
    rnn.embedding === nothing && error("rnn has no embedding!")
    if all(length.(batch).==length(batch[1]))
        return _forw(rnn,cat(batch...;dims=2)',h...;o...)
    end
    if !sorted
        v   = sortperm(batch;by=length,rev=true,alg=MergeSort)
        rev = sortperm(v;alg=MergeSort)
        batch= batch[v]
    end
    tokens,bsizes = b2bs(batch)
    y,h,c,_ = _forw(rnn,tokens,h...;batchSizes=bsizes,o...)
    inds    = bs2ind(bsizes);
    if !sorted
        inds=inds[rev];
        h!=nothing && (h=h[:,rev,:]) #Knet lacks
        c!=nothing && (c=c[:,rev,:]) #Knet lacks
    end
    return y,h,c,inds
end

function _forw(rnn::RNN,x,h...;o...)
    if rnn.embedding === nothing
        y,h,c,_ = rnnforw(rnn.specs,rnn.params,x,h...;o...)
    else
        y,h,c,_ = rnnforw(rnn.specs,rnn.params,rnn.embedding(x),h...;o...)
    end
    return y,h,c,nothing
end

function b2bs(batch::Vector{Vector{T}}) where T<:Integer
    tokens = Int[]
    bsizes = Int[]
    B      = length(batch)
    Lmax   = length(first(batch))
    for t = 1:Lmax
        bs = 0
        for k = 1:B
            if t<=length(batch[k])
                push!(tokens,batch[k][t])
                bs += 1
            end
        end
        push!(bsizes,bs)
    end
    return tokens,bsizes
end

function bs2inds(batchSizes)
    B = batchSizes[1]
    inds = Any[]
    for i=1:B
        ind = i.+cumsum(filter(x->(x>=i),batchSizes)[1:end-1])
        push!(inds,append!(Int[i],ind))
    end
    return inds
end
