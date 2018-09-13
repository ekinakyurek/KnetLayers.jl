struct ReLU <: Model
end
(l::ReLU)(x) = relu.(x)

struct Sigm <: Model
end
(l::Sigm)(x) = sigm.(x)

struct Tanh <: Model
end
(l::Tanh)(x) = tanh.(x)

struct LogP <: Model
end
(l::LogP)(x;dims=:) = logp(x;dims=dims)

struct SoftMax <: Model
end
(l::SoftMax)(x;dims=:) = exp.(logp(x;dims=dims))

struct LogSumExp <: Model
end
(l::LogSumExp)(x;dims=:) = logsumexp(x;dims=dims)

struct ELU <: Model
end
(l::ELU)(x) = relu.(x) + (exp.(min.(0,x)) .- one(eltype(x)))

struct LeakyReLU <: Model
    α::AbstractFloat
    LeakyReLU(alpha::AbstractFloat=0.2) = new(alpha)
end

struct Dropout <: Model
    p::Real
    Dropout(p::Real=0.0) = new(p)
end
(l::Dropout)(x) = dropout(x,l.p)
