include("header.jl")

@testset "rnn" begin
    x    = zeros(10,1)
    ind  = rand(1:10)
    x[ind,1] = 1.0
    l = LSTM(10,5;embed=5)
    @test all(l([ind]).y .== l(x).y)
    l = SRNN(10,5;embed=5)
    @test all(l([ind]).y .== l(x).y)
    l = GRU(10,5;embed=5)
    @test all(l([ind]).y .== l(x).y)
end
