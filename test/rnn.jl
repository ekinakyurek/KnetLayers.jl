include("header.jl")

@testset "rnn" begin
    arrtype = KnetLayers.arrtype
    x    = arrtype(zeros(10,1))
    ind  = rand(1:10)
    x[ind,1] = 1.0
    l = LSTM(input=10,hidden=5,embed=5)
    @test all(l([ind]).y .== l(x).y)
    l = SRNN(input=10,hidden=5,embed=5)
    @test all(l([ind]).y .== l(x).y)
    l = GRU(input=10,hidden=5,embed=5)
    @test all(l([ind]).y .== l(x).y)
end
