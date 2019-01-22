####
#### Output Structure
####
"""
    struct RNNOutput
        y
        hidden
        memory
        indices
    end

Outputs of the RNN models are always `RNNOutput`
`hidden`,`memory` and `indices` may be nothing depending on the kwargs you used in forward.

`y` is last hidden states of each layer. `size(y)=(H/2H,[B,T])`.
If you use unequal length instances in a batch input, `y` becomes 2D array `size(y)=(H/2H,sum_of_sequence_lengths)`.
See `indices` and `PadRNNOutput` to get correct time outputs for a specific instance or to pad whole output.

`h` is the hidden states in each timesstep. `size(h) = h,B,L/2L`

`c` is the hidden states in each timesstep. `size(h) = h,B,L/2L`

`indices` is corresponding instace indices for your `RNNOutput.y`. You may call `yi = y[:,indices[i]]`.
"""
struct RNNOutput{T,V,Z}
    y::T
    hidden::V
    memory::Z
    indices::Union{Vector{Vector{Int}},Nothing}
end

####
#### RNN Types
####
"""
    SRNN(;input=inputSize, hidden=hiddenSize, activation=:relu, options...)
    LSTM(;input=inputSize, hidden=hiddenSize, options...)
    GRU(;input=inputSize, hidden=hiddenSize, options...)

    (1) (l::T)(x; kwargs...) where T<:AbstractRNN
    (2) (l::T)(x::Array{Int}; batchSizes=nothing, kwargs...) where T<:AbstractRNN
    (3) (l::T)(x::Vector{Vector{Int}}; sorted=false, kwargs...) where T<:AbstractRNN

All RNN layers has above forward run(1,2,3) functionalities.

(1) `x` is an input array with size equals d,[B,T]

(2) For this You need to have an RNN with embedding layer.
`x` is an integer array and inputs coressponds one hot vector indices.
You can give 2D array for minibatching as rows corresponds to one instance.
You can give 1D array with minibatching by specifying batch batchSizes argument.
Checkout `Knet.rnnforw` for this.

(3) For this You need to have an RNN with embedding layer.
`x` is an vector of integer vectors. Every integer vector corresponds to an
instance. It automatically batches inputs. It is better to give inputs as sorted.
If your inputs sorted you can make `sorted` argument true to increase performance.

see RNNOutput

# options

* `embed=nothing`: embedding size or and embedding layer
* `numLayers=1`: Number of RNN layers.
* `bidirectional=false`: Create a bidirectional RNN if `true`.
* `dropout=0`: Dropout probability. Ignored if `numLayers==1`.
* `skipInput=false`: Do not multiply the input with a matrix if `true`.
* `dataType=eltype(KnetLayers.arrtype)`: Data type to use for weights. Default is Float32.
* `algo=0`: Algorithm to use, see CUDNN docs for details.
* `seed=0`: Random number seed for dropout. Uses `time()` if 0.
* `winit=xavier`: Weight initialization method for matrices.
* `binit=zeros`: Weight initialization method for bias vectors.
* `usegpu=(KnetLayers.arrtype <: KnetArray)`: GPU used by default if one exists.

# Keywords

* hx=nothing : initial hidden states
* cx=nothing : initial memory cells
* hy=false   : if true returns h
* cy=false   : if true returns c

"""
AbstractRNN

for layer in (:SRNN, :LSTM, :GRU)
    layername=string(layer)
    @eval begin
        mutable struct $layer <: AbstractRNN
            embedding::Union{Nothing,Embed}
            params
            specs::Knet.RNN
            gatesview::Dict
        end

        (m::$layer)(x,h...;o...) = RNNOutput(_forw(m,x,h...;o...)...)

        function $layer(;input::Integer, hidden::Integer, embed=nothing, activation=:relu,
                         usegpu=(arrtype <: KnetArray), dataType=eltype(arrtype), o...)
            embedding,inputSize = _getEmbed(input,embed)
            rnnType = $layer==SRNN ? activation : Symbol(lowercase($layername))
            r = Knet.RNN(inputSize, hidden; rnnType=rnnType, usegpu=usegpu, dataType=dataType, o...)
            gatesview  = Dict{Symbol,Any}()
            for l in 1:r.numLayers, d in 0:r.direction
                for (g,id) in gate_mappings($layer)
                    for (ih,ihid) in input_mappings, (ty,param) in param_mappings
                         gatesview[Symbol(ty,ih,g,l,d)] = rnnparam(r, (r.direction+1)*(l-1)+d+1, id[ihid], param; useview=true)
                    end
                end
            end
            $layer(embedding,r.w,r,gatesview)
        end
    end
end
gate_mappings(::Type{SRNN}) = Dict(:h=>(1,2))
gate_mappings(::Type{GRU})  = Dict(:r=>(1,4),:u=>(2,5),:n=>(3,6))
gate_mappings(::Type{LSTM}) = Dict(:i=>(1,5),:f=>(2,6),:n=>(3,7),:o=>(4,8))
const input_mappings = Dict(:i=>1,:h=>2)
const param_mappings = Dict(:w=>1,:b=>2)

####
#### Utils
####
"""

    PadSequenceArray(batch::Vector{Vector{T}}) where T<:Integer

Pads a batch of integer arrays with zeros

```
julia> PadSequenceArray([[1,2,3],[1,2],[1]])
3×3 Array{Int64,2}:
 1  2  3
 1  2  0
 1  0  0
 ```

"""
function PadSequenceArray(batch::Vector{Vector{T}}; pad=0) where T<:Integer
    B      = length(batch)
    lngths = length.(batch)
    Tmax   = maximum(lngths)
    padded = Array{T}(undef,B,Tmax)
    @inbounds for n = 1:B
        padded[n,1:lngths[n]] = batch[n]
        padded[n,lngths[n]+1:end] .= pad
    end
    return padded
end


"""

    PadRNNOutput(s::RNNOutput)
Pads a rnn output if it is produces by unequal length batches
`size(s.y)=(H/2H,sum_of_sequence_lengths)` becomes `(H/2H,B,Tmax)`

"""
function PadRNNOutput(s::RNNOutput)
    s.indices == nothing && return s,nothing
    d = size(s.y,1)
    B = length(s.indices)
    lngths = length.(s.indices)
    Tmax = maximum(lngths)
    mask = trues(d,B,Tmax)
    cw = []
    @inbounds for i=1:B
        y1 = s.y[:,s.indices[i]]
        df = Tmax-lngths[i]
        if df > 0
            mask[:,:,end-df+1:end] .= false
            cpad = zeros(Float32,d*df) # zeros(Float32,2d,df)
            kpad = atype(cpad)
            ypad = reshape(cat1d(y1,kpad),d,Tmax) # hcat(y1,kpad)
            push!(cw,ypad)
        else
            push!(cw,y1)
        end
    end
    RNNOutput(reshape(vcat(cw...),d,B,Tmax),s.hidden,s.memory,nothing),mask
end

function _pack_sequence(batch::Vector{Vector{T}}) where T<:Integer
    tokens = Int[]
    bsizes = Int[]
    B      = length(batch)
    Lmax   = length(first(batch))
    @inbounds for t = 1:Lmax
        bs = 0
        @inbounds for k = 1:B
            if t<=length(batch[k])
                push!(tokens,batch[k][t])
                bs += 1
            end
        end
        push!(bsizes,bs)
    end
    return tokens,bsizes
end

function _batchSizes2indices(batchSizes)
    B = batchSizes[1]
    inds = Vector{Int}[]
    for i=1:B
        ind = i.+cumsum(filter(x->(x>=i),batchSizes)[1:end-1])
        push!(inds,append!(Int[i],ind))
    end
    return inds
end

_getEmbed(input::Int,embed::Nothing) = (nothing,input)
_getEmbed(input::Int,embed::Embed)   = size(embed.w,2) == input ? (embed,input) : error("dimension mismatch in your embedding")
_getEmbed(input::Int,embed::Integer) = (Embed(input=input,output=embed),embed)

function _forw(rnn::AbstractRNN,seq::Array{T},h...;batchSizes=nothing,o...) where T<:Integer
    rnn.embedding === nothing && error("rnn has no embedding!")
    ndims(seq) == 1 && batchSizes === nothing && (seq = reshape(seq,1,length(seq)))
    y,hidden,memory,_ = rnnforw(rnn.specs,rnn.params,rnn.embedding(seq),h...;batchSizes=batchSizes,o...)
    return y,hidden,memory,nothing
end

function _forw(rnn::AbstractRNN,batch::Vector{Vector{T}},h...;sorted=true,o...) where T<:Integer
    rnn.embedding === nothing && error("rnn has no embedding!")
    if all(length.(batch).==length(batch[1]))
        return _forw(rnn,cat(batch...;dims=2)',h...;o...)
    end
    if !sorted
        v   = sortperm(batch;by=length,rev=true,alg=MergeSort)
        rev = sortperm(v;alg=MergeSort)
        batch= batch[v]
    end
    tokens,bsizes = _pack_sequence(batch)
    y,hidden,memory,_  = _forw(rnn,tokens,h...;batchSizes=bsizes,o...)
    inds    = _batchSizes2indices(bsizes);
    if !sorted
        inds=inds[rev];
        hidden!=nothing && (hidden=_sort3D(hidden,rev))
        memory!=nothing && (memory=_sort3D(memory,rev))
    end
    y,hidden,memory,inds
end

function _forw(rnn::AbstractRNN,x,h...;o...)
    if rnn.embedding === nothing
        y,hidden,memory,_ = rnnforw(rnn.specs,rnn.params,x,h...;o...)
    else
        y,hidden,memory,_ = rnnforw(rnn.specs,rnn.params,rnn.embedding(x),h...;o...)
    end
    y,hidden,memory,nothing
end

_sort3D(hidden::Array,inds) = hidden[:,inds,:]
function _sort3D(h::KnetArray,inds)
    container = [];
    for i=1:size(h,3)
        push!(container,h[:,:,i][:,inds])
    end
    reshape(cat1d(container...),size(h))
end
