"""
    CrossEntropyLoss(dims=1)
    (l::CrossEntropyLoss)(scores, answers)
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
    dims::Int
end
CrossEntropyLoss(;dims=1) = CrossEntropyLoss(dims)
(l::CrossEntropyLoss)(y,answers::Array{<:Integer})=nll(y,answers;dims=l.dims)

"""
    BCELoss(average=true)
    (l::BCELoss)(scores, answers)
    Computes binary cross entropy given scores(predicted values) and answer labels. answer values should be {0,1}, then it returns negative of
    mean|sum(answers * log(p) + (1-answers)*log(1-p)) where p is equal to 1/(1 + exp.(scores)). See also LogisticLoss.
"""
struct BCELoss <: Loss
end
(l::BCELoss)(y,answers::Array{<:Integer})=bce(y,answers)

"""
    LogisticLoss(average=true)
    (l::LogisticLoss)(scores, answers)
    Computes logistic loss given scores(predicted values) and answer labels. answer values should be {-1,1}, then it returns mean|sum(log(1 +
    exp(-answers*scores))). See also `BCELoss`.
"""
struct LogisticLoss <: Loss
end
(l::LogisticLoss)(y,answers::Array{<:Integer})=logistic(y,answers)
