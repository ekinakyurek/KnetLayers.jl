include("header.jl")

@testset "mlp" begin
     x = randn(10,2)
     m = MLP(10,5,2;winit=randn,binit=zeros)
     y = m(x)
     @test true
end
