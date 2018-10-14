include("header.jl")

@testset "cnn" begin

    arrtype = KnetLayers.arrtype
    
    @testset "conv" begin
        m = Conv(height=3,width=3,channels=3,filters=5,stride=1,padding=1,mode=1)
        x = arrtype(zeros(10,10,3,2))
        y = m(x)
        @test size(y) == (10,10,5,2)
    end
end
