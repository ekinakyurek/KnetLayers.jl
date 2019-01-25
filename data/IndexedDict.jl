import Base: get, length, getindex, push!, append!
""""
    IndexedDict{T}

A data structure for vocabulary in NLP Models, or any other categorical inputs
details: http://www.ekinakyurek.com/indexeddict/

```JULIA
julia> data = "this is an example data with example words"
"this is an example data with example words"

julia> vocab = IndexedDict{String}()
IndexedDict{String}(Dict{String,Int64}(), String[])

julia> append!(vocab,split(data))
IndexedDict{String}(Dict("this"=>1,"is"=>2,"example"=>4,"data"=>5,"words"=>7,"with"=>6,"an"=>3), ["this", "is", "an", "example", "data", "with", "words"])

julia> vocab["example"]
4

julia> vocab[4]
"example"

julia> vocab[[1,2,3,4]]
4-element Array{String,1}:
 "this"
 "is"
 "an"
 "example"

julia> vocab[["example","this"]]
2-element Array{Int64,1}:
 4
 1

julia> push!(vocab,"a-new-word")
IndexedDict{String}(Dict("a-new-word"=>8,"this"=>1,"is"=>2,"example"=>4,"data"=>5,"words"=>7,"with"=>6,"an"=>3), ["this", "is", "an", "example", "data", "with", "words", "a-new-word"])

julia> vocab["a-new-word"]
8

julia> length(vocab)
8
```
"""
struct IndexedDict{T}
    toIndex::Dict{T,Int};
    toElement::Vector{T};

    IndexedDict{T}(toIndex,toElement) where T = new(toIndex,toElement)
    IndexedDict{T}(toIndex,toElement) where T<:Integer = error("Cannot Create IndexedDict of Integers")
    IndexedDict{T}(toIndex,toElement) where T<:AbstractArray = error("Cannot Create IndexedDict of Arrays")
end

IndexedDict{T}() where T = IndexedDict{T}(Dict{T,Int}(),T[])

function IndexedDict(toElement::Vector{T}) where T
    toIndex=Dict{T,Int}(v=>k for (k,v) in enumerate(toElement))
    IndexedDict(toIndex, toElement)
end

function IndexedDict(toIndex::Dict{T,Int}) where T
    toElement=Vector{T}(undef,length(toIndex))
    for (k,v) in toIndex; toElement[v]=k; end
    IndexedDict{T}(toIndex,toElement)
end

get(d::IndexedDict,v,default) = get(d.toIndex,v,default)
length(d::IndexedDict) = length(d.toElement)
getindex(d::IndexedDict,inds::Integer) = d.toElement[inds]
getindex(d::IndexedDict{T},inds::T) where T = d.toIndex[inds]
getindex(d::IndexedDict{T}, elements::Array{T,1}) where T = map(e->d[e], elements)
getindex(d::IndexedDict, inds::Array{<:Integer}) = d.toElement[inds]
append!(d1::IndexedDict{T}, d2::IndexedDict{T}) where T = append!(d1,d2.toElement)

function push!(d::IndexedDict, element)
    if !haskey(d.toIndex,element)
        d.toIndex[element]=length(d)+1;
        push!(d.toElement,element)
    end
    return d
end

function append!(d::IndexedDict, elements)
     for element in elements
          push!(d,element)
     end
     return d
end
