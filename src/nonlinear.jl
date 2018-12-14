"""
    ReLU()
    (l::ReLU)(x) = max(0,x)

Rectified Linear Unit function.
"""
struct ReLU <: Activation end
(l::ReLU)(x) = relu.(x)

"""
    Sigm()
    (l::Sigm)(x) = sigm(x)

Sigmoid function
"""
struct Sigm <: Activation end
(l::Sigm)(x) = sigm.(x)

"""
    Tanh()
    (l::Tanh)(x) = tanh(x)

Tangent hyperbolic function
"""
struct Tanh <: Activation end
(l::Tanh)(x) = tanh.(x)


"""
    ELU()
    (l::ELU)(x) = elu(x) -> Computes x < 0 ? exp(x) - 1 : x

Exponential Linear Unit nonlineariy.
"""
struct ELU <: Activation end
(l::ELU)(x) = elu.(x)

"""
    LeakyReLU(α=0.2)
    (l::LeakyReLU)(x) -> Computes x < 0 ? α*x : x
"""
mutable struct LeakyReLU <: Activation
    α::AbstractFloat
    LeakyReLU(alpha::AbstractFloat=0.2) = new(alpha)
end
(l::LeakyReLU)(x) = relu.(x) .+ l.α*min.(0,x)

"""
    Dropout(p=0)

Dropout Layer. `p` is the droput probability.
"""
mutable struct Dropout <: Activation
    p::Real
end
Dropout(;p=0) = Dropout(p)
(l::Dropout)(x; enable=true) = enable ? dropout(x,l.p) : x

"""
    LogSoftMax(dims=:)
    (l::LogSoftMax)(x)

Treat entries in x as as unnormalized log probabilities and return normalized log probabilities.

dims is an optional argument, if not specified the normalization is over the whole x, otherwise the normalization is performed over the given dimensions. In
particular, if x is a matrix, dims=1 normalizes columns of x and dims=2 normalizes rows of x.
"""
struct LogSoftMax <: Activation
    dims::Union{Integer,Colon}
end
LogSoftMax(;dims=:) = LogSoftMax(dims)
(l::LogSoftMax)(x) = logp(x;dims=l.dims)

"""
    SoftMax(dims=:)
    (l::SoftMax)(x)

Treat entries in x as as unnormalized scores and return softmax probabilities.

dims is an optional argument, if not specified the normalization is over the whole x, otherwise the normalization is performed over the given dimensions. In
particular, if x is a matrix, dims=1 normalizes columns of x and dims=2 normalizes rows of x.
"""
struct SoftMax <: Activation
    dims::Union{Integer,Colon}
end
SoftMax(;dims=:) = SoftMax(dims)
(l::SoftMax)(x) = softmax(x;dims=l.dims)

"""
    LogSumExp(dims=:)
    (l::LogSumExp)(x)

  Compute log(sum(exp(x);dims)) in a numerically stable manner.

  dims is an optional argument, if not specified the summation is over the whole x, otherwise the summation is performed over the given dimensions. In particular if x
  is a matrix, dims=1 sums columns of x and dims=2 sums rows of x.
"""
struct LogSumExp <: Activation
    dims::Union{Integer,Colon}
end
LogSumExp(;dims=:) = LogSumExp(dims)
(l::LogSumExp)(x) = logsumexp(x;dims=l.dims)
