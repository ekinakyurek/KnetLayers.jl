"""
    Embed(inputSize,embedSize;winit=xavier)


Creates and embedding layer according to given `inputSize` and `embedSize`.

By default embedding parameters initialized with xavier,
you can change it `winit` argument


    (m::Embed)(x::Array{T}) where T<:Integer
    (m::Embed)(x)


Embed objects are callable with an input which is either and integer array
(one hot encoding) or an N-dimensional matrix. For N-dimensional matrix,
`size(x,1)==inputSize`

"""
struct Embed <: Model
    w
end
Embed(input::Int,embed::Int;winit=xavier) = Embed(Prm(winit(embed,input)))
(m::Embed)(x::Array{T}) where T<:Integer = m.w[:,x]
function (m::Embed)(x)
    if ndims(x) > 2
        y =  m.w * reshape(x,size(x,1),prod(size(x)[2:end]))
        return reshape(y,size(y,1),size(x)[2:end]...)
    else
        return m.w * x
    end
end

"""
    Linear(inputSize,outputSize;kwargs...)


Creates and linear layer according to given `inputSize` and `outputSize`.

By default embedding parameters initialized with xavier,
you can change it `winit` argument

# Keywords

* `winit=xavier`: weight initialization distribution
* `bias=zeros`: bias initialization distribution

    (m::Linear)(x)
"""

struct Linear <: Model
    w
    b
end
Linear(i::Int,o::Int;winit=xavier,binit=zeros)=Linear(Prm(winit(o,i)),Prm(binit(o)))
(m::Linear)(x) = (m.w * x .+ m.b)

"""
    Dense(inputSize,outputSize;kwargs...)

Creates and deense layer according to given `inputSize` and `outputSize`.

# Keywords

* `winit=xavier`: weight initialization distribution
* `bias=zeros`: bias initialization distribution
* `f=ReLU()`: activation function

    (m::Dense)(x)
"""

struct Dense <: Model
    w
    b
    f
end
Dense(i::Int,o::Int;f=ReLU(),winit=xavier,binit=zeros)=Dense(Prm(winit(o,i)),Prm(binit(o)),f)
(m::Dense)(x) = m.f((m.w * x .+ m.b))

"""
    Conv(h,[w,c,o];kwargs...)

Creates and convolutional layer according to given filter dimensions.

# Keywords

* `winit=xavier`: weight initialization distribution
* `bias=zeros`: bias initialization distribution
* `padding=0`: the number of extra zeros implicitly concatenated at the start and at the end of each dimension.
* `stride=1`: the number of elements to slide to reach the next filtering window.
* `upscale=1`: upscale factor for each dimension.
* `mode=0`: 0 for convolution and 1 for cross-correlation.
* `alpha=1`: can be used to scale the result.
* `handle`: handle to a previously created cuDNN context. Defaults to a Knet allocated handle.

    (m::Conv)(x)
"""

struct Conv <: Model
    w
    b
    padding
    stride
    upscale
    mode
    alpha
end
Conv(h::Int;winit=xavier,binit=zeros,opts...)=Conv(Prm(winit(h,1,1,1)),binit(1,1,1,1);opts...)
Conv(h::Int,w::Int;winit=xavier,binit=zeros,opts...)=Conv(Prm(winit(h,w,1,1)),binit(1,1,1,1);opts...)
Conv(h::Int,w::Int,c::Int;winit=xavier,binit=zeros,opts...)=Conv(Prm(winit(h,w,c,1)),binit(1,1,1,1);opts...)
Conv(h::Int,w::Int,c::Int,o::Int;winit=xavier,binit=zeros,opts...)=Conv(Prm(winit(h,w,c,o)),binit(1,1,o,1);opts...)
Conv(w,b;stride=1,padding=0,mode=0,upscale=1,alpha=1) = Conv(w,b,padding,stride,upscale,mode,alpha)
function (m::Conv)(x)
     n = ndims(x)
     if n == 4
         return conv4(m.w,x;stride=m.stride,padding=m.padding,mode=m.mode,upscale=m.upscale,alpha=m.alpha) .+ m.b
     elseif n == 3
         x1 = reshape(x,size(x)...,1)
     elseif n == 2
         x1 = reshape(x,size(x)...,1,1)
     elseif n == 1
         x1 = reshape(x,size(x)...,1,1,1)
     else
         error("Conv supports 1,2,3,4 D arrays only")
     end
     y = conv4(m.w,x1;stride=m.stride,padding=m.padding,mode=m.mode,upscale=m.upscale,alpha=m.alpha) .+ m.b
     return reshape(y,size(y)[1:n])
end

struct BatchNorm <: Model
    params
    moments::Knet.BNMoments
end
BatchNorm(channels::Int;o...) =BatchNorm(Prm(bnparams(eltype(atype),channels)),bnmoments(;o...))
(m::BatchNorm)(x;o...) = batchnorm(x,m.moments,m.params;o...)
