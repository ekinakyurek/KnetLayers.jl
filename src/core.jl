abstract type Model end
"""
    Projection(inputSize,embedSize;winit=xavier)

Creates a projection layer according to given `inputSize` and `embedSize`.

    (m::Projection)(x) = m.w*x

By default parameters initialized with xavier, you can change it with `winit` argument
"""
struct Projection <: Model
    w
end
Projection(input::Int,embed::Int;winit=xavier) = Projection(Prm(winit(embed,input)))
(m::Projection)(x::Array{T}) where T<:Integer = m.w[:,x]
function (m::Projection)(x;keepsize=true)
    if ndims(x) > 2
        s = size(x)
        y =  m.w * reshape(x,s[1],prod(s[2:end]))
        keepsize ? reshape(y,size(y,1),s[2:end]...) : y
    else
        return m.w * x
    end
end

"""
    Embed(inputSize,embedSize;winit=xavier)


Creates an embedding layer according to given `inputSize` and `embedSize`.

By default embedding parameters initialized with xavier,
you can change it `winit` argument


    (m::Embed)(x::Array{T}) where T<:Integer
    (m::Embed)(x; keepsize=true)


Embed objects are callable with an input which is either and integer array
(one hot encoding) or an N-dimensional matrix. For N-dimensional matrix,
`size(x,1)==inputSize`

"""
Embed=Projection

"""
    Linear(inputSize,outputSize;kwargs...)
    (m::Linear)(x; keepsize=true) #forward run


Creates and linear layer according to given `inputSize` and `outputSize`.

By default embedding parameters initialized with xavier,
you can change it `winit` argument

# Keywords

* `winit=xavier`: weight initialization distribution
* `bias=zeros`: bias initialization distribution

"""
struct Linear <: Model
    w::Projection
    b
end
Linear(i::Int,o::Int;winit=xavier,binit=zeros)=Linear(Projection(i,o;winit=winit),Prm(binit(o)))
(m::Linear)(x;keepsize=true) = m.w(x;keepsize=keepsize) .+ m.b


"""
    Dense(inputSize,outputSize;kwargs...)
    (m::Dense)(x; keepsize=true) #forward run

Creates and deense layer according to given `inputSize` and `outputSize`.

# Keywords

* `winit=xavier`: weight initialization distribution
* `bias=zeros`: bias initialization distribution
* `f=ReLU()`: activation function
* `keepsize=true`: if false ndims(y)=2 all dimensions other than first one
squeezed to second dimension

"""
struct Dense <: Model
    l::Linear
    f
end
Dense(i::Int,o::Int;f=ReLU(),winit=xavier,binit=zeros)=Dense(Linear(i,o;winit=winit,binit=binit),f)
(m::Dense)(x;keepsize=true) = m.f(m.l(x;keepsize=keepsize))

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
struct BatchNorm <: Model
    params
    moments::Knet.BNMoments
end
BatchNorm(channels::Int;o...) =BatchNorm(Prm(bnparams(eltype(atype),channels)),bnmoments(;o...))
(m::BatchNorm)(x;o...) = batchnorm(x,m.moments,m.params;o...)
