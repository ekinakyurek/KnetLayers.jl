include("header.jl")

@testset "loss" begin

    arrtype = KnetLayers.arrtype;
    x = arrtype(randn(10,10))
    indices = rand(1:10,10)
    l1 = CrossEntropyLoss(dims=2)
    l1(x,indices);
    @test true

    x = arrtype(rand(10))
    indices = rand([0,1],10)
    l2 = BCELoss()
    l2(x,indices)
    @test true

    x = arrtype(randn(10))
    indices = rand([-1,1],10)
    l3 = LogisticLoss()
    l3(x,indices)
    @test true

end
