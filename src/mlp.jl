struct MLP <: Model
     layers::Tuple{Vararg{Linear}}
end
MLP(h::Int...;winit=xavier,binit=zeros)=MLP(Linear.(h[1:end-1],h[2:end];winit=winit,binit=binit))
function (m::MLP)(x;activation=relu,prob=0)
    for layer in m.layers
        x = layer(dropout(x,prob))
        if layer !== m.layers[end]
            x = activation.(x)
        end
    end
    return x
end
