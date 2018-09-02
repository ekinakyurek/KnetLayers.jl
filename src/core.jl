struct Embed <: Model
    w
end
Embed(input::Int,embed::Int;winit=xavier) = Embed(param(winit(embed,input)))
(m::Embed)(x::Array{T}) where T<:Integer = m.w[:,x]
function (m::Embed)(x)
    if ndims(x) > 2
        y =  m.w * reshape(x,size(x,1),prod(size(x)[2:end]))
        return reshape(y,size(y,1),size(x)[2:end]...)
    else
        return m.w * x
    end
end

struct Linear <: Model
    w
    b
end
Linear(i::Int,o::Int;winit=xavier,binit=zeros)=Linear(param(winit(o,i)),param(binit(o)))
(m::Linear)(x) = (m.w * x .+ m.b)

struct Conv <: Model
    w
    b
end
Conv(h::Int;winit=xavier,binit=zeros)=Conv(param(winit(h,1,1,1)),binit(1,1,1,1))
Conv(h::Int,w::Int;winit=xavier,binit=zeros)=Conv(param(winit(h,w,1,1)),binit(1,1,1,1))
Conv(h::Int,w::Int,c::Int;winit=xavier,binit=zeros)=Conv(param(winit(h,w,c,1)),binit(1,1,1,1))
Conv(h::Int,w::Int,c::Int,o::Int;winit=xavier,binit=zeros)=Conv(param(winit(h,w,c,o)),binit(1,1,o,1))
function (m::Conv)(x;o...)
     n = ndims(x)
     if n == 4
         return conv4(m.w,x;o...) .+ m.b
     elseif n == 3
         y = conv4(m.w,reshape(x,size(x)...,1);o...) .+ m.b
     elseif n == 2
         y = conv4(m.w,reshape(x,size(x)...,1,1);o...) .+ m.b
     elseif n == 1
         y = conv4(m.w,reshape(x,size(x)...,1,1,1);o...) .+ m.b
     else
         error("Conv supports 1,2,3,4 D arrays only")
     end
     return reshape(y,size(y)[1:n])
end

struct BatchNorm <: Model
    params
    moments::BNMoments
end
BatchNorm(channels::Int;o...) =BatchNorm(param(bnparams(eltype(atype),channels)),bnmoments(;o...))
(m::BatchNorm)(x;o...) = batchnorm(x,m.moments,m.params;o...)
