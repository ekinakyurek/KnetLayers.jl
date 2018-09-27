"""
    CrossEntropyLoss(; dims=1)
    (l::CrossEntropyLoss)(scores, answers::Array{<:Integer})
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
struct CrossEntropyLoss <: Model
    dims::Int
end
CrossEntropyLoss(;dims=1) = CrossEntropyLoss(dims)
(l::CrossEntropyLoss)(y,answers::Array{<:Integer})=nll(y,answers;dims=l.dims)
