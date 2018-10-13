"""
    Multiply(input=inputDimension, output=outputDimension, winit=WeightInitialization)

Creates a matrix multiplication layer based on `inputDimension` and `outputDimension`.
    (m::Multiply) = m.w * x

By default parameters initialized with xavier, you may change it with `winit` argument.

# Keywords
* `input=inputDimension`: input dimension
* `output=outputDimension`: output dimension
* `winit=xavier`: weight initialization distribution
* `bias=zeros`: bias initialization distribution
"""

struct Multiply <: Layer
    w
end

Multiply(;input::Int, output::Int, winit=xavier, o...) =  Multiply(param(output, input, init=winit, o...))

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
    Embed(input=inputSize,output=embedSize, winit=xavier)
Creates an embedding layer according to given `inputSize` and `embedSize` where `inputSize` is your number of unique items you want to embed, and `embedSize` is the size of output vectors.
By default parameters initialized with xavier, you yam change it with `winit` argument.

    (m::Embed)(x::Array{T}) where T<:Integer
    (m::Embed)(x; keepsize=true)

    
Embed objects are callable with an input which is either and integer array
(one hot encoding) or an N-dimensional matrix. For N-dimensional matrix,
`size(x,1)==inputSize`

"""

Embed = Multiply

# TODO: find a better documentation style e.g put input : and output etc.
# Input: Type of its input is an ::Array{T} where T<: Integer


"""
    Linear(input=inputSize, output=outputSize, winit=xavier, binit=zeros)

Creates and linear layer according to given `inputSize` and `outputSize`.

# Keywords
* `input=inputSize`   input dimension
* `output=outputSize` output dimension
* `winit=xavier`: weight initialization distribution
* `bias=zeros`: bias initialization distribution
"""

struct Linear <: Layer
    w::Multiply
    b
end

function Linear(;input::Int, output::Int, winit=xavier, binit=zeros, o...)
    return Linear(Multiply(input=input, output=output, winit=winit, o...),
                  param(output, init=binit, o...))
end

(m::Linear)(x) = m.w(x) .+ m.b


"""
    Dense(input=inputSize, output=outputSize, winit=xavier, binit=zeros, activation="relu")

Creates and deense layer according to given `input=inputSize` and `output=outputSize`.
If activation is not provided acts like a Linear Layer.
# Keywords
* `winit=xaiver`: weight initialization distribution
* `bias=zeros`:   bias initialization distribution
* `activation=relu`  activation function
"""

struct Dense <: Layer
    l::Linear
    f
end

function Dense(;input::Int, output::Int, activation=nothing, winit=xavier, binit=zeros, o...)

    if activation == nothing
        return Linear(input=input, output=output, winit=winit, binit=binit, o...)
    end
    
    activation = (typeof(activation) == Symbol ? eval(activation) : eval(Symbol(activation)))
    return Dense(Linear(input=input, output=output, winit=winit, binit=binit, o...), activation)
end

(m::Dense)(x) = m.f.(m.l(x))



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
* `varinit=ones`: The function used for initialize the runn
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

BatchNorm(channels::Int;o...) =BatchNorm(Prm(bnparams(eltype(atype),channels)),bnmoments(;o...))
(m::BatchNorm)(x;o...) = batchnorm(x,m.moments,m.params;o...)
