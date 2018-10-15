using KnetLayers

leftpad(p::Int,x::Array)=cat(p*ones(Int,size(x,1)),x;dims=2);
rightpad(x::Array,p::Int)=cat(x,p*ones(Int,size(x,1));dims=2);

struct S2S
    encoder
    decoder
    output
    loss
end

model = S2S(LSTM(input=11,hidden=128,embed=9), 
            LSTM(input=11,hidden=128,embed=9),
            Multiply(input=128,output=11),CrossEntropyLoss()) 

(m::S2S)(x)     = m.output(m.decoder(leftpad(10,sort(x,dims=2)), m.encoder(x;hy=true).hidden).y)
predict(m,x)    = getindex.(argmax(Array(m(x)), dims=1)[1,:,:], 1)
loss(m,x,ygold) = m.loss(m(x),ygold)

dataxy(x) = (x,rightpad(sort(x, dims=2), 11))
B, maxL= 64, 15; # Batch size and maximum sequence length for training
data = [dataxy([rand(1:9) for j=1:B, k=1:rand(1:maxL)]) for i=1:10000]
train!(model,data;loss=loss,optimizer=Adam())


"julia> predict(model,[3 2 1 4 5 9 3 5 6 6 1 2 5 7 8 5 5 2 9;])
1×19 Array{Int64,2}:
 1  1  2  2  2  3  3  4  5  5  5  5  5  6  6  7  8  9  9

 julia> sort([3 2 1 4 5 9 3 5 6 6 1 2 5 7 8 5 5 2 9];dims=2)
1×19 Array{Int64,2}:
 1  1  2  2  2  3  3  4  5  5  5  5  5  6  6  7  8  9  9
"
