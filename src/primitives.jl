"""
    Multiply(input=inputDimension, output=outputDimension, winit=xavier, atype=KnetLayers.arrtype)

Creates a matrix multiplication layer based on `inputDimension` and `outputDimension`.
    (m::Multiply) = m.w * x

By default parameters initialized with xavier, you may change it with `winit` argument.

# Keywords
* `input=inputDimension`: input dimension
* `output=outputDimension`: output dimension
* `winit=xavier`: weight initialization distribution
* `atype=KnetLayers.arrtype` : array type for parameters. 
   Default value is KnetArray{Float32} if you have gpu device. Otherwise it is Array{Float32}
"""
struct Multiply <: Layer
    w
end

Multiply(;input::Int, output::Int, winit=xavier, atype=arrtype) =  Multiply(param(output, input; init=winit, atype=atype))

(m::Multiply)(x::Array{T}) where T<:Integer = m.w[:,x] # Lookup (EmbedLayer)

# TODO: Find a faster (or compound) way for tensor-product
function (m::Multiply)(x; keepsize=true)
    if ndims(x) > 2 
        s = size(x)
        y = m.w * reshape(x, s[1], prod(s[2:end]))
        return (keepsize ? reshape(y, size(y, 1), s[2:end]...) : y)
    else
        return m.w * x
    end
end


"""
    Embed(input=inputSize, output=embedSize, winit=xavier, atype=KnetLayers.arrtype)
Creates an embedding layer according to given `inputSize` and `embedSize` where `inputSize` is your number of unique items you want to embed, and `embedSize` is the size of output vectors.
By default parameters initialized with xavier, you yam change it with `winit` argument.

    (m::Embed)(x::Array{T}) where T<:Integer
    (m::Embed)(x; keepsize=true)

    
Embed objects are callable with an input which is either and integer array
(one hot encoding) or an N-dimensional matrix. For N-dimensional matrix,
`size(x,1)==inputSize`

# Keywords

* `input=inputDimension`: input dimension
* `output=embeddingDimension`: output dimension
* `winit=xavier`: weight initialization distribution
* `atype=KnetLayers.arrtype` : array type for parameters. 
   Default value is KnetArray{Float32} if you have gpu device. Otherwise it is Array{Float32}

"""
Embed = Multiply

# TODO: find a better documentation style e.g put input : and output etc.
# Input: Type of its input is an ::Array{T} where T<: Integer


"""
    Linear(input=inputSize, output=outputSize, winit=xavier, binit=zeros, atype=KnetLayers.arrtype)

Creates and linear layer according to given `inputSize` and `outputSize`.

# Keywords
* `input=inputSize`   input dimension
* `output=outputSize` output dimension
* `winit=xavier`: weight initialization distribution
* `bias=zeros`: bias initialization distribution
* `atype=KnetLayers.arrtype` : array type for parameters. 
   Default value is KnetArray{Float32} if you have gpu device. Otherwise it is Array{Float32}
"""
struct Linear <: Layer
    w::Multiply
    b
end

function Linear(;input::Int, output::Int, winit=xavier, binit=zeros, atype=arrtype)
    Linear(Multiply(input=input, output=output, winit=winit, atype=atype),param(output, init=binit, atype=atype))
end

(m::Linear)(x) = m.w(x) .+ m.b


"""
    Dense(input=inputSize, output=outputSize, activation=relu, winit=xavier, binit=zeros, atype=KnetLayers.arrtype)

Creates and deense layer according to given `input=inputSize` and `output=outputSize`.
If activation is `nothing`, it acts like a `Linear` Layer.
# Keywords
* `input=inputSize`   input dimension
* `output=outputSize` output dimension
* `winit=xaiver`: weight initialization distribution
* `bias=zeros`:   bias initialization distribution
* `activation=ReLU()`  activation function(it does not broadcast) or an  activation layer
* `atype=KnetLayers.arrtype` : array type for parameters.
   Default value is KnetArray{Float32} if you have gpu device. Otherwise it is Array{Float32}
"""
struct Dense <: Layer
    l::Linear
    activation
end

function Dense(;input::Int, output::Int, activation=ReLU(), winit=xavier, binit=zeros, atype=arrtype)
    activation == nothing && return Linear(input=input, output=output, winit=winit, binit=binit, atype=atype)
    Dense(Linear(input=input, output=output, winit=winit, binit=binit, atype=atype), activation)
end

(m::Dense)(x) = m.activation(m.l(x))



"""
    BatchNorm(channels:Int;options...)
    (m::BatchNorm)(x;training=false) #forward run
# Options
* `momentum=0.1`: A real number between 0 and 1 to be used as the scale of
 last mean and variance. The existing running mean or variance is multiplied by
 (1-momentum).
* `mean=nothing': The running mean.
* `var=nothing`: The running variance.
* `meaninit=zeros`: The function used for initialize the running mean. Should either be `nothing` or
of the form `(eltype, dims...)->data`. `zeros` is a good option.
* `varinit=ones`: The function used for initialize the run
* `elementtype=eltype(KnetLayers.arrtype)` : element type ∈ {Float32,Float64} for parameters. Default value is `eltype(KnetLayers.arrtype)`

# Keywords
* `training`=nothing: When training is true, the mean and variance of x are used and moments
 argument is modified if it is provided. When training is false, mean and variance
 stored in the moments argument are used. Default value is true when at least one
 of x and params is AutoGrad.Value, false otherwise.
"""
struct BatchNorm <: Layer
    params
    moments::Knet.BNMoments
end

BatchNorm(channels::Int;elementtype=eltype(arrtype),o...) =BatchNorm(bnparams(elementtype,channels),bnmoments(;o...))
(m::BatchNorm)(x;o...) = batchnorm(x,m.moments,m.params;o...)
