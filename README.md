# KLayers

KLayers provides configurable deep learning layers for Knet, fostering your model development.

## Example Usages

```JULIA  
using Knet, KLayers
#Instantiate an MLP model with random parameters
mlp = MLP(100,50,20;activation=sigm) # input size=100, hidden=50 and output=20
#Do a prediction
prediction = mlp(randn(100,1))

#Instantiate Conv layer with random parameters
cnn = Conv(3,3,20,10;padding=1,stride=1) # A conv filter with H=3,W=3,C=20,O=10
#Filter your input
output = cnn(randn(224,224,3,1))


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
y,h,c,_   = lstm([1,2,3,4],[5,6];sorted=true,hy=true,cy=true)
#To see and modify rnn params in a structured view
lstm.gatesview

```

## Exported Layers
```
Linear
Embed
MLP
Conv
LSTM
GRU
SRNN
```

## TO-DO
1) Enhance `Conv` Interface   
2) `CNN` model  
3) Export `Pool`,`Unpool`,`DeConv`,`Dropout` and non-linear functions(`relu`,`elu`,...) as layers.  
4) Known layers such Google's `inception`   
5) Known embeddings such `Gloove`   
6) Pretrained Models   
