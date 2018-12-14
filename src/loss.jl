using Statistics
"""
    CrossEntropyLoss(dims=1)
    (l::CrossEntropyLoss)(scores, answers; average=true)
Calculates negative log likelihood error on your predicted scores.
`answers` should be integers corresponding to correct class indices.
If an answer is 0, loss from that answer will not be included.
This is usefull feature when you are working with unequal length sequences.

if dims==1
* size(scores) = C,[B,T1,T2,...]
* size(answers)= [B,T1,T2,...]
elseif dims==2
* size(scores) = [B,T1,T2,...],C
* size(answers)= [B,T1,T2,...]
"""
struct CrossEntropyLoss <: Loss
    dims::Integer
end
CrossEntropyLoss(;dims=1) = CrossEntropyLoss(dims)
(l::CrossEntropyLoss)(y,answers::Array{<:Integer}; average=true) = nllmask(y, answers; dims=l.dims, average=average)

"""
    BCELoss(average=true)
    (l::BCELoss)(scores, answers)
    Computes binary cross entropy given scores(predicted values) and answer labels. answer values should be {0,1}, then it returns negative of
    mean|sum(answers * log(p) + (1-answers)*log(1-p)) where p is equal to 1/(1 + exp.(scores)). See also LogisticLoss.
"""
struct BCELoss <: Loss end
(l::BCELoss)(y,answers::Array{<:Integer})=bce(y,answers)

"""
    LogisticLoss(average=true)
    (l::LogisticLoss)(scores, answers)
    Computes logistic loss given scores(predicted values) and answer labels. answer values should be {-1,1}, then it returns mean|sum(log(1 +
    exp(-answers*scores))). See also `BCELoss`.
"""
struct LogisticLoss <: Loss end
(l::LogisticLoss)(y,answers::Array{<:Integer})=logistic(y,answers)

####
#### Utils
####
function nllmask(y,a::AbstractArray{<:Integer}; dims=1, average=true)
    indices = findindices(y, a, dims=dims)
    lp = logp(y,dims=dims)[indices]
    average ? -mean(lp) : -sum(lp)
end

function findindices(y,a::AbstractArray{<:Integer}; dims=1)
    n       = length(a)
    nonmask = a .> 0
    indices = Vector{Int}(undef,sum(nonmask))
    if dims == 1                   # instances in first dimension
        y1 = size(y,1)
        y2 = div(length(y),y1)
        if n != y2; throw(DimensionMismatch()); end
        k = 1
        @inbounds for (j,v) in enumerate(nonmask)
            !v && continue
            indices[k] = (j-1)*y1 + a[j]
            k += 1
        end
    elseif dims == 2               # instances in last dimension
        y2 = size(y,ndims(y))
        y1 = div(length(y),y2)
        if n != y1; throw(DimensionMismatch()); end
        k = 1
        @inbounds for (j,v) in enumerate(nonmask)
            !v && continue
            indices[k] = (a[j]-1)*y1 + j
            k += 1
        end
    else
        error("findindices only supports dims = 1 or 2")
    end
    return indices
end
