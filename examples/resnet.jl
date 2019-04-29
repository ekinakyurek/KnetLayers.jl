using Statistics, Knet, KnetLayers, BSON, ImageMagick, Images

"""
ResNet Models

Pre trained ResNet{18,34,50,101,152} weights are available.
See below to see how to use it!

```julia
  julia> include(KnetLayers.dir("examples","resnet.jl"))

  julia> m = ResNet{50}(trained=true)
  ResNet{50}()

  julia> topK(m.labels, m("/Users/ekin/Downloads/gray-wolf_sam-parks.png");K=5)
  5-element Array{String,1}:
  "timber wolf, grey wolf, gray wolf, Canis lupus"
  "white wolf, Arctic wolf, Canis lupus tundrarum"
  "red wolf, maned wolf, Canis rufus, Canis niger"
  "dingo, warrigal, warragal, Canis dingo"
  "coyote, prairie wolf, brush wolf, Canis latrans"
```
"""
struct ResNet{N}
  layers::Chain
  labels::Array{String}
end

struct BasicV2
  layers::Chain
  downsample::Union{Conv,Nothing}
end

function BasicV2(channels, stride; downsample=false, in_channels=0, kwargs...)
  layers = Chain(
                BatchNorm(in_channels),
                ReLU(),
                Conv(height=3, width=3, inout=in_channels=>channels,
                     padding = 1, stride = stride, binit=nothing, mode=1),
                BatchNorm(channels÷4),
                ReLU(),
                Conv(height=3, width=3, inout=channels=>channels,
                     padding = 1, stride = 1, binit=nothing, mode=1)
                )
  if downsample
    return BasicV2(layers, Conv(height=1, width=1, inout=in_channels=>channels,
                                padding = 0, stride = stride, mode=1, binit=nothing))
  else
    return BasicV2(layers,nothing)
  end
end

function (m::BasicV2)(x)
      residual = x
      x = m.layers[1:2](x)
      if m.downsample !== nothing
          residual = m.downsample(x)
      end
      x = m.layers[3:length(m.layers.layers)](x)
      return x + residual
end

struct BottleneckV2
  layers::Chain
  downsample::Union{Conv,Nothing}
end

function BottleneckV2(channels, stride; downsample=false, in_channels=0, kwargs...)
  layers = Chain(
                BatchNorm(in_channels),
                ReLU(),
                Conv(height=1, width=1, inout=in_channels=>channels÷4,
                     padding = 0, stride = 1, binit=nothing, mode=1),
                BatchNorm(channels÷4),
                ReLU(),
                Conv(height=3, width=3, inout=channels÷4=>channels÷4,
                     padding = 1, stride = stride, binit=nothing, mode=1),
                BatchNorm(channels÷4),
                ReLU(),
                Conv(height=1, width=1, inout=channels÷4=>channels,
                     padding = 0, stride = 1, binit=nothing, mode=1)
                )
  if downsample
    return BottleneckV2(layers,
                        Conv(height=1, width=1, inout=in_channels=>channels,
                        padding = 0, stride = stride, mode=1, binit=nothing))
  else
    return BottleneckV2(layers,nothing)
  end
end

function (m::BottleneckV2)(x)
      residual = x
      x = m.layers[1:2](x)
      if m.downsample !== nothing
          residual = m.downsample(x)
      end
      x = m.layers[3:length(m.layers.layers)](x)
      return x + residual
end

function _make_layer(block, layers, channels, stride, stage_index; in_channels=0)
  layer = [block(channels, stride, downsample=(channels != in_channels), in_channels=in_channels)]
  for _ in 1:layers-1
         push!(layer, block(channels, 1, downsample=false, in_channels=channels))
  end
  return Chain(layer...)
end

function _init(block, layers, channels; classes=1000, N=50, stage=0)
  @assert length(layers) == length(channels) - 1 "error"
  top = Chain(BatchNorm(3),
                Conv(height=7, width=7, inout=3=>channels[1],
                     padding = 1, stride = 2, binit=nothing, mode=1),
                BatchNorm(channels[1]), ReLU(),
                Pool(window=3, padding = 1, stride = 2))

    stages = Chain[]
    in_channels = channels[1]
    for (i, num_layer) in enumerate(layers)
          stride = i == 1 ? 1 : 2
          push!(stages,_make_layer(block, num_layer, channels[i+1], stride, i+1, in_channels=in_channels))
          in_channels = channels[i+1]
          stage==i && return Chain(top,Chain(stages...))
    end
    bottom =  Chain(BatchNorm(channels[end]), ReLU(), Pool(window=(7,7), mode=1))
    stage==5 && return Chain(top,Chain(stages...), bottom)
    classifier = Chain(mat,Linear(input=in_channels,output=classes))
    return Chain(top,Chain(stages...),bottom,classifier)
end

configs = Dict(18  => (BasicV2, [2, 2, 2, 2], [64, 64, 128, 256, 512]),
               34  => (BasicV2, [3, 4, 6, 3], [64, 64, 128, 256, 512]),
               50  => (BottleneckV2, [3, 4, 6, 3], [64, 256, 512, 1024, 2048]),
               101 => (BottleneckV2, [3, 4, 23, 3],[64, 256, 512, 1024, 2048]),
               152 => (BottleneckV2, [3, 8, 36, 3],[64, 256, 512, 1024, 2048]))

@inline (m::ResNet)(x) = m.layers(x)

(m::ResNet)(x::Union{AbstractMatrix,AbstractString}) where {T,N} = m(preprocess(x))
topK(labels::Vector{String},y;K=5) = labels[sortperm(vec(y);rev=true)[1:K]]

function ResNet{N}(;trained=false, stage=0, mfile=KnetLayers.dir("examples","resnet$(N)v2.bson")) where N
   resnet = ResNet{N}(_init(configs[N]...; stage=stage),getLabels())
   if trained
     if !isfile(mfile)
        download(mfile,mfile) #FIXME
     end
     weights = BSON.load(mfile)
     loadResNet!(resnet,weights)
   end
   return resnet
end

Base.show(io::IO, ::ResNet{N}) where N = print(io, "ResNet{$N}()")

###
#### Utils
###
const atype = KnetLayers.arrtype
transfer!(p::Param, x)  = transfer!(p.value,x)
transfer!(p::KnetArray, x::AbstractArray) = p .= KnetArray(x)
transfer!(p,x) =  p .= x
to4D(x) = reshape(convert(atype,x),1,1,length(x),1)
toArrType(x) = convert(atype,x)

function getLabels(labels=KnetLayers.dir("examples","imagenet_labels.txt"))
  if !isfile(labels)
      download("https://github.com/ekinakyurek/KnetLayers.jl/releases/download/v0.2.0/imagenet_labels.txt",labels) #FIXME
  end
  return readlines(labels)
end

import Base: /
/(a::RGB, b::RGB) = RGB(a.r/b.r, a.g/b.g, a.b/b.b)

function preprocess(img::AbstractMatrix)
  # Resize such that smallest edge is 256 pixels long
    img = resize_smallest_dimension(img, 256)
    im  = (center_crop(img, 224) .- RGB(0.485, 0.456, 0.406)) ./ RGB(0.229, 0.224, 0.225)
    z   =  channelview(im)
    z1  = Float32.(permutedims(z, (3, 2, 1))[:,:,:,:]);
end

preprocess(img::AbstractString)  = preprocess(RGB.(load(img)))

# Resize an image such that its smallest dimension is the given length
function resize_smallest_dimension(im, len)
  reduction_factor = len/minimum(size(im)[1:2])
  new_size = size(im)
  new_size = (
      round(Int, size(im,1)*reduction_factor),
      round(Int, size(im,2)*reduction_factor),
  )
  if reduction_factor < 1.0
    # Images.jl's imresize() needs to first lowpass the image, it won't do it for us
    im = imfilter(im, KernelFactors.gaussian(0.75/reduction_factor), Inner())
  end
  return imresize(im, new_size)
end

# Take the len-by-len square of pixels at the center of image `im`
function center_crop(im, len)
  l2 = div(len,2)
  adjust = len % 2 == 0 ? 1 : 0
  return im[div(end,2)-l2:div(end,2)+l2-adjust,div(end,2)-l2:div(end,2)+l2-adjust]
end


# Load Utils
function loadBNLayer!(m::BatchNorm, weights, prefix)
  m.moments.var  = weights[Symbol(prefix*"running_var")]  |> to4D
  m.moments.mean = weights[Symbol(prefix*"running_mean")] |> to4D
  m.params       = vcat(weights[Symbol(prefix*"gamma")],weights[Symbol(prefix*"beta")]) |> toArrType
end

function loadConvLayer!(m::Conv, weights, prefix; binit=false)
  transfer!(m.weight, weights[Symbol(prefix*"weight")])
  if binit
    transfer!(m.bias.b,  weights[Symbol(prefix*"bias")])
  end
end

function loadDenseLayer!(m::Linear, weights, prefix; binit=true)
  transfer!(m.mult.weight, permutedims(weights[Symbol(prefix*"weight")],(2,1)))
  if binit
    transfer!(m.bias.b,  weights[Symbol(prefix*"bias")])
  end
end

function loadTop!(top::Chain, weights, prefix)
  loadBNLayer!(top[1], weights, string(prefix,"batchnorm0_"))
  loadConvLayer!(top[2], weights, string(prefix,"conv0_"))
  loadBNLayer!(top[3], weights, string(prefix,"batchnorm1_"))
end

function loadBottom!(bottom::Chain, weights, prefix)
  loadBNLayer!(bottom[1], weights, string(prefix,"batchnorm2_"))
end

function loadFinal!(bottom::Chain, weights, prefix)
  loadDenseLayer!(bottom[2], weights, string(prefix,"dense0_"))
end

function loadBlock!(block::BottleneckV2, weights, prefix, bn, conv)
  loadBNLayer!(block.layers[1], weights, string(prefix,"batchnorm$(bn)_"))
  loadConvLayer!(block.layers[3], weights, string(prefix,"conv$(conv)_"))
  bn+=1; conv+=1
  loadBNLayer!(block.layers[4], weights, string(prefix,"batchnorm$(bn)_"))
  loadConvLayer!(block.layers[6], weights, string(prefix,"conv$(conv)_"))
  bn+=1; conv+=1
  loadBNLayer!(block.layers[7], weights, string(prefix,"batchnorm$(bn)_"))
  loadConvLayer!(block.layers[9], weights, string(prefix,"conv$(conv)_"))
  bn+=1; conv+=1
  if block.downsample !== nothing
    loadConvLayer!(block.downsample, weights, string(prefix,"conv$(conv)_"))
    conv+=1
  end
  return bn, conv
end

function loadBlock!(block::BasicV2, weights, prefix, bn, conv)
  loadBNLayer!(block.layers[1], weights, string(prefix,"batchnorm$(bn)_"))
  loadConvLayer!(block.layers[3], weights, string(prefix,"conv$(conv)_"))
  bn+=1; conv+=1
  loadBNLayer!(block.layers[4], weights, string(prefix,"batchnorm$(bn)_"))
  loadConvLayer!(block.layers[6], weights, string(prefix,"conv$(conv)_"))
  bn+=1; conv+=1
  if block.downsample !== nothing
    loadConvLayer!(block.downsample, weights, string(prefix,"conv$(conv)_"))
    conv+=1
  end
  return bn, conv
end

function loadStage!(stage::Chain, weights, prefix)
  bn   = 0
  conv = 0
  for (i,block) in enumerate(stage.layers)
      bn, conv = loadBlock!(block,weights,prefix,bn,conv)
  end
end

const idmap = Dict{Int,Int}(18=>2,34=>3,50=>4,101=>5,152=>7)
function loadResNet!(resnet::ResNet{N}, weights; prefix="resnetv2") where N
  prefix=prefix*string(idmap[N],"_");
  loadTop!(resnet.layers[1],weights,prefix)
  for (i,stage) in enumerate(resnet.layers[2])
      loadStage!(resnet.layers[2][i], weights, "$(prefix)stage$(i)_")
  end
  if length(resnet.layers) > 2
    loadBottom!(resnet.layers[3],weights,prefix)
  end
  if length(resnet.layers) > 3
    loadFinal!(resnet.layers[4],weights,prefix)
  end
  return resnet
end
