include("header.jl")

@testset "primitive" begin

    @testset "Multiply" begin
        inputDim = 10; outputDim = 2;
        m = Multiply(input=inputDim, output=outputDim; winit=randn)
        atype = typeof(Knet.value(m.w))
        x = atype(zeros(inputDim, 1))
        y1 = m(x)
        y2 = atype(zeros(outputDim, 1))
        @test y1==y2
    end


    @testset "Embed" begin
        m  = Embed(input=10, output=2, winit=randn)
        atype = typeof(Knet.value(m.w))
        x    = atype(zeros(10,1))
        ind  = rand(1:10)
        x[ind,1] = 1.0
        y1 = m(x)
        y2 = m([ind])
        @test y1==y2
    end

    @testset "linear" begin
        inputDim=10; outputDim=3
        m = Linear(input=inputDim, output=outputDim; winit=randn,binit=zeros)
        atype = typeof(Knet.value(m.w))
        x = atype(zeros(10,2))
        y = m(x)
        @test true
    end

    @testset "dense" begin
        inputDim=10; outputDim=3
        m = Dense(input=inputDim, output=outputDim; winit=randn,binit=zeros)
        atype = typeof(Knet.value(m.w))
        x = atype(zeros(10,2))
        y = m(x)
        @test true
    end

    @testset "conv" begin
        m = Conv(3,3,3,5;stride=1,padding=1,mode=1)
        atype = typeof(Knet.value(m.w))
        atype = (gpu() > 0 ? KnetArray{Float32}: Array{Float32})
        x = atype(zeros(10,10,3,2))
        y = m(x)
        @test size(y) == (10,10,5,2)
    end

end
