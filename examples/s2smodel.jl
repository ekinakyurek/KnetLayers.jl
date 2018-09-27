using KnetLayers
B, maxL= 32, 15; # Batch size and maximum sequence length for training
pad(y::Array{<:Integer})=(y[:,1].=10;y) # insert pad chracter for T=0

struct S2S; encoder; decoder; output; loss; end # model consists of encoder-decoder
(m::S2S)(x)     = m.output(m.decoder(pad(sort(x,dims=2)), m.encoder(x;hy=true).hidden ).y)
predict(m,x)    = getindex.(argmax(Array(m(x)),dims=1)[1,:,:],1)
loss(m,x,ygold) = m.loss(m(x),ygold)

dataxy(x) = (x,sort(x,dims=2))
data = [dataxy([rand(1:9) for j=1:B,k=1:rand(1:maxL)]) for i=1:10000]

model = S2S(LSTM(10,32;embed=5),LSTM(10,32;embed=5),Projection(32,10),CrossEntropyLoss())
train!(model,data;loss=loss,optimizer=Adam(;lr=.0001))
predict(model,[3 2 1 4 5 9 3 5 6 6 1 2 5;])
