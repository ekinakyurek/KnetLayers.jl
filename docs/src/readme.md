# KnetLayers
KnetLayers provides configurable deep learning layers for Knet, fostering your model development. You are able to use Knet and AutoGrad functionalities without adding them to current workspace.

## Example Usages
```JULIA
using KnetLayers
#Instantiate an MLP model with random parameters
mlp = MLP(100,50,20; activation=Sigm()) # input size=100, hidden=50 and output=20
#Do a prediction
prediction = mlp(randn(Float32,100,1))

#Instantiate Conv layer with random parameters
cnn = Conv(height=3,width=3,channels=3,filters=10;padding=1,stride=1) # A conv layer
#Filter your input
output = cnn(randn(Float32,224,224,3,1))

#Instantiate an LSTM model
lstm = LSTM(input=100,hidden=100,embed=50)

#You can use integers to represent one hot vectors
#For example a pass over 5-Length sequence
rnnoutput = lstm([3,2,1,4,5];hy=true,cy=true)

rnnoutput.y
rnnoutput.hidden
rnnoutput.memory

#You can also use normal array inputs for low-level control
#One iteration of LSTM
rnnoutput = lstm(randn(100,1);hy=true,cy=true)

#Pass over a 10-length sequence:
rnnoutput = lstm(randn(100,1,10);hy=true,cy=true)

#Pass over a mini-batch data which includes unequal length sequences
rnnoutput = lstm([[1,2,3,4],[5,6]];sorted=true,hy=true,cy=true)

#To see and modify rnn params in a structured view
lstm.gatesview
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
NonLinear:
  Sigm, Tanh, ReLU, ELU
  LogSoftMax, LogSumExp, SoftMax
  Dropout
```

## TO-DO
1) Use dropout only in training
2) Loss functions
3) Examples
4) Special layers such Google's `inception`   
5) Known embeddings such `Gloove`   
6) Pretrained Models   
