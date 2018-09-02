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
