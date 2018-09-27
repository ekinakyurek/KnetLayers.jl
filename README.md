# KnetLayers
KnetLayers provides configurable deep learning layers for Knet, fostering your model development. You can use Knet and AutoGrad functionalities without adding them to current workspace.

## Example Usages
```JULIA  
using KnetLayers
#Instantiate an MLP model with random parameters
mlp = MLP(100,50,20) # input size=100, hidden=50 and output=20
#Do a prediction
prediction = mlp(randn(Float32,100,1);activation=sigm) #defaul activation is relu

#Instantiate Conv layer with random parameters
cnn = Conv(3,3,3,10;padding=1,stride=1) # A conv filter with H=3,W=3,C=3,O=10
#Filter your input
output = cnn(randn(Float32,224,224,3,1))

#Instantiate an LSTM model
lstm = LSTM(100,100;embed=50) #input size=100, hidden=100, embedding=50
#You can use integers to represent one hot vectors
#For example a pass over 5-Length sequence
y,h,c,_ = lstm([3,2,1,4,5];hy=true,cy=true)
#You can also use normal array inputs for low-level control
#One iteration of LSTM
y,h,c,_ = lstm(randn(100,1);hy=true,cy=true)
#Pass over a 10-length sequence:
y,h,c,_ = lstm(randn(100,1,10);hy=true,cy=true)
#Pass over a mini-batch data which includes unequal length sequences
y,h,c,_ = lstm([1,2,3,4],[5,6];sorted=true,hy=true,cy=true)
#To see and modify rnn params in a structured view
lstm.gatesview

```

## Exported Layers
```
Core:
  Linear, Embed, Conv, Dense
MLP
RNN:
  LSTM, GRU, SRNN
NonLinear:
  Sigm,Tanh,ReLU,ELU
  LogP, LogSumExp, SoftMax
  Dropout
```

## TO-DO
1) Use dropout only in training
2) Loss functions
3) Examples
4) Special layers such Google's `inception`   
5) Known embeddings such `Gloove`   
6) Pretrained Models   
