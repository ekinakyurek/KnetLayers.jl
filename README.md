# KnetLayers

[![](https://img.shields.io/badge/docs-latest-blue.svg)](https://ekinakyurek.github.io/KnetLayers.jl/latest)
[![](https://gitlab.com/JuliaGPU/KnetLayers/badges/master/pipeline.svg)](https://gitlab.com/JuliaGPU/KnetLayers/pipelines)
[![](https://travis-ci.org/ekinakyurek/KnetLayers.jl.svg?branch=master)](https://travis-ci.org/ekinakyurek/KnetLayers.jl)
[![codecov](https://codecov.io/gh/ekinakyurek/KnetLayers.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/ekinakyurek/KnetLayers.jl)

KnetLayers provides configurable deep learning layers for Knet, fostering your model development. You are able to use Knet and AutoGrad functionalities without adding them to current workspace.

## How It Is
```Julia
model = Chain(
          Dense(input=768, output=128, activation=Sigm()),
          Dense(input=128, output=10, activation=nothing),
          CrossEntropyLoss()
        )

loss(x, y) = model[end](model[1:end-1](x), y)
```

## Example Layer Usages
```JULIA
using KnetLayers

#Instantiate an MLP model with random parameters
mlp = MLP(100,50,20; activation=Sigm()) # input size=100, hidden=50 and output=20

#Do a prediction with the mlp model
prediction = mlp(randn(Float32,100,1))

#Instantiate a convolutional layer with random parameters
cnn = Conv(height=3, width=3, inout=3=>10, padding=1, stride=1) # A conv layer

#Filter your input with the convolutional layer
output = cnn(randn(Float32,224,224,3,1))

#Instantiate an LSTM model
lstm = LSTM(input=100, hidden=100, embed=50)

#You can use integers to represent one-hot vectors.
#Each integer corresponds to vocabulary index of corresponding element in your data.

#For example a pass over 5-Length sequence
rnnoutput = lstm([3,2,1,4,5];hy=true,cy=true)

#After you get the output, you may acces to hidden states and
#intermediate hidden states produced by the lstm model
rnnoutput.y
rnnoutput.hidden
rnnoutput.memory

#You can also use normal array inputs for low-level control
#One iteration of LSTM with a random input
rnnoutput = lstm(randn(100,1);hy=true,cy=true)

#Pass over a random 10-length sequence:
rnnoutput = lstm(randn(100,1,10);hy=true,cy=true)

#Pass over a mini-batch data which includes unequal length sequences
rnnoutput = lstm([[1,2,3,4],[5,6]];sorted=true,hy=true,cy=true)

#To see and modify rnn params in a structured view
lstm.gatesview
```

## Example Model

An example of sequence to sequence models which learns sorting integer numbers.
```JULIA
using KnetLayers

struct S2S # model definition
    encoder
    decoder
    output
    loss
end
# initialize model
model = S2S(LSTM(input=11,hidden=128,embed=9),
            LSTM(input=11,hidden=128,embed=9),
            Multiply(input=128,output=11),
            CrossEntropyLoss())

# Helper functions for padding
leftpad(p::Int,x::Array)=cat(p*ones(Int,size(x,1)),x;dims=2)
rightpad(x::Array,p::Int)=cat(x,p*ones(Int,size(x,1));dims=2)

# forward functions
(m::S2S)(x)     = m.output(m.decoder(leftpad(10,sort(x,dims=2)), m.encoder(x;hy=true).hidden).y)
predict(m,x)    = getindex.(argmax(Array(m(x)), dims=1)[1,:,:], 1)
loss(m,x,ygold) = m.loss(m(x),ygold)

# create sorting data
# 10 is used as start token and 11 is stop token.
dataxy(x) = (x,rightpad(sort(x, dims=2), 11))
B, maxL= 64, 15; # Batch size and maximum sequence length for training
data = [dataxy([rand(1:9) for j=1:B, k=1:rand(1:maxL)]) for i=1:10000]

#train your model
train!(model,data;loss=loss,optimizer=Adam())

"julia> predict(model,[3 2 1 4 5 9 3 5 6 6 1 2 5;])
1×14 Array{Int64,2}:
 1  2  2  2  3  4  4  5  5  5  6  7  9  11

julia> sort([3 2 1 4 5 9 3 5 6 6 1 2 5;];dims=2)
1×13 Array{Int64,2}:
 1  1  2  2  3  3  4  5  5  5  6  6  9
"
```

## Exported Layers
```
Core:
  Multiply, Linear, Embed, Dense
CNN
  Conv, DeConv, Pool, UnPool
MLP
RNN:
  LSTM, GRU, SRNN
Loss:
  CrossEntropyLoss, BCELoss, LogisticLoss
NonLinear:
  Sigm, Tanh, ReLU, ELU
  LogSoftMax, LogSumExp, SoftMax,
  Dropout
```

## TO-DO
3) Examples
4) Special layers such Google's `inception`   
5) Known embeddings such `Gloove`   
6) Pretrained Models   
