"""
    one_hot(shape::Vararg{Int}, a::AbstractArray{<:Integer}; dims=1, atype=arrtype)
        Creates one hot vectors given shape and gold indices.

    one_hot(y:AbstractArray, a::AbstractArray{<:Integer}; dims=1, atype=arrtype)
        Creates one hot vectors given array and gold indices.


    #Example
    ```julia
        julia> KnetLayers.one_hot((5,3),ones(Int,3); dims=1)
            5×3 Array{Float32,2}:
             1.0  1.0  1.0
             0.0  0.0  0.0
             0.0  0.0  0.0
             0.0  0.0  0.0
             0.0  0.0  0.0
    ```
"""
function one_hot(shape:: NTuple{N, Int}, goldindices::AbstractArray{<:Integer}; dims=1, atype=KnetLayers.arrtype) where N
    T = eltype(atype)
    labels = zeros(T,shape)
    linearinds = findindices(labels,goldindices, dims=dims)
    @inbounds for i in linearinds
        labels[i] = one(T)
    end
    return convert(atype,labels)
end

one_hot(y, a::AbstractArray{<:Integer}; atype=typeof(value(y)), dims=1) = one_hot(size(y),a;atype=atype,dims=dims)
