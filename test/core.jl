include("header.jl")

@testset "core" begin

    @testset "embed" begin
        x    = zeros(10,1)
        ind  = rand(1:10)
        x[ind,1] = 1.0
        m  = Embed(10,2;winit=randn)
        y1 = m(x)
        y2 = m([ind])
        @test y1==y2
    end

    @testset "linear" begin
         x = zeros(10,2)
         m = Linear(10,3;winit=randn,binit=zeros)
         y = m(x)
         @test true
    end

    @testset "dense" begin
        x = zeros(10,2)
        m = Dense(10,3;f=Sigm(),winit=randn,binit=zeros)
        y = m(x)
        @test true
    end

    @testset "conv" begin
        x = zeros(10,10,3,2)
        m = Conv(3,3,3,5;stride=1,padding=1,mode=1)
        y = m(x)
        @test size(y) == (10,10,5,2)
    end
end
