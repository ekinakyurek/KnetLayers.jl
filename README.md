# KLayers

KLayers provides configurable deep learning layers for Knet, fostering your model development.

## Example Usages

```JULIA  
using Knet, KLayers
#Instantiate an MLP model with random parameters
mlp = MLP(100,50,20;activation=sigm) # input size=100, hidden=50 and output=20
#Do a prediction
prediction = mlp(randn(100,1))

lstm = LSTM(100,100;embed=50) #input size=100, hidden=100, embedding=50
#Do a prediction
y,h,c,_ = lstm([1,2,3,4]) #you can use integers as one hot vector
#Do a prediction with a mini-batch which includes unequal length sequences
y,h,c   = lstm([1,2,3,4],[5,6];sorted=true)
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
