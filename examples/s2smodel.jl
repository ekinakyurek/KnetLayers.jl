using KnetLayers

"""
julia> predict(model,[3 2 1 4 5 9 3 5 6 6 1 2 5;])
1×14 Array{Int64,2}:
 1  2  2  2  3  4  4  5  5  5  6  7  9  11
"""
struct S2S; encoder; decoder; output; loss; end

# Initialize model
model = S2S(LSTM(input=11,hidden=128,embed=9),
            LSTM(input=11,hidden=128,embed=9),
            Multiply(input=128,output=11),
            CrossEntropyLoss())

# Forward functions
(m::S2S)(x) = m.output(m.decoder(pad(10,sort(x,dims=2)), m.encoder(x; hy=true).hidden).y)
(m::S2S)(x,ygold) = m.loss(m(x),ygold)
predict(m::S2S,x) = getindex.(argmax(Array(m(x)), dims=1)[1,:,:], 1)

# Helper functions for padding
pad(p::Int,x::Array; dims=2) = cat(fill(p,size(x,1)),x;dims=dims)
pad(x::Array,p::Int; dims=2) = cat(x,fill(p,size(x,1));dims=dims)

# Create sorting data: 10 is used as start token and 11 is stop token.
dataxy(x) = (x,pad(sort(x, dims=2), 11))
batchSize, maxLength, dataLength= 64, 15, 2000; # Batch size and maximum sequence length for training
data = [dataxy([rand(1:9) for j=1:batchSize, k=1:rand(1:maxLength)]) for i=1:dataLength]

# Train your model
progress!(adam(model,data))
# Test the model
@show predict(model,[3 2 1 4 5 9 3 5 6 6 1 2 5;])
