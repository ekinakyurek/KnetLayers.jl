
module GPT

using  KnetLayers, WordTokenizers, BytePairEncoding, JSON, NPZ
import KnetLayers: IndexedDict, AbstractTransformer, Layer, arrtype, dir, Datasets
using KnetLayers.Datasets
import KnetLayers.Datasets: Container

struct Gpt <: AbstractTransformer
    ts::Chain
    drop::Dropout
    loss::CrossEntropyLoss
end

function Gpt(size::Int, head::Int, ps::Int, layer::Int;
             max_len::Int = 512, trainable = true, act = ELU(), pdrop = 0.1)
    rem(size, head) != 0 && error("size not divisible by head")
    Gpt(size, head, div(size, head), ps, layer; max_len=max_len, trainable=trainable, act=act, pdrop=pdrop)
end

function Gpt(size::Int, head::Int, hs::Int, ps::Int, layer::Int;
             max_len::Int = 512, trainable = true, act = ELU(), pdrop = 0.1)
    Gpt(Chain([Transformer(size, head, hs, ps; future=false, act=act, pdrop=pdrop) for i = 1:layer]...),
        Dropout(pdrop), CrossEntropyLoss())
end

function (gpt::Gpt)(x, mask=nothing; all::Bool=false)
    e = gpt.drop(x)
    t = gpt.ts(e)
    if mask != nothing
        mask = convert(arrtype,mask)
        t = t .* reshape(mask,1,size(mask)...)
    end
    return t
end
"""
    lmloss(embed, onehot, encoding, mask)
compute the language modeling loss for Gpt, onehot is the onehot array of the origin
input sentence. encoding the output of Gpt, mask is the mask between input sentences.
"""
lmloss(embed::Embed, o, t, mask) = lmloss(embed, o, t, mask)
function lmloss(embed::Embed, et, t, mask)
    N = ndims(t)
    if N == 3
        t = t[:, 1:end-1, :]
        s = size(t)
        #FIXME: transpose
        out = Embed(embed.weight')
        return KnetLayers.nllmask(out(t), et[2:end,:].*mask[2:end,:]) #(vocab, seq_len*batch)
    elseif N == 2
        t = t[:, 1:end-1]
        s = size(t)
        out = Embed(embed.weight')
        return KnetLayers.nllmask(out(t), et[2:end,:] .*mask[2:end]) #(vocab, seq_len*batch)
    end
end

function Base.show(io::IO, gpt::Gpt)
    hs = div(size(gpt.ts[1].mh.iqproj)[1], gpt.ts[1].mh.head)
    h, ps = size(gpt.ts[1].pw.layers[end])
    print(io, "Gpt(")
    print(io, "layers=$(length(gpt.ts.layers)), ")
    print(io, "head=$(gpt.ts[1].mh.head), ")
    print(io, "head_size=$(hs), ")
    print(io, "pwffn_size=$(ps), ")
    print(io, "size=$(h))")
end


function load_gpt_pretrain_params()
    shapes  = JSON.parsefile(dir("examples/gpt/pretrain/params_shapes.json"))
    offsets = accumulate(+, prod.(shapes))
    shapes  = map(s -> length(s) > 1 ? (s[end], s[end-1]) : s, shapes)
    params  = cat([npzread(joinpath(dirname(@__FILE__), "pretrain/params_$(i).npy")) for i = 0:9]..., dims=1)
    params  = [collect(reshape(selectdim(params, 1, a+1:b), s...)) for (a, b, s) in zip([0;offsets[1:end-1]], offsets, shapes)]
    return params
end

gpt_tokenizer(x) = toktok_tokenize(text_standardize(x))

function load_gpt_pretrain(n::Int=12;
                           startsym="_start_",
                           delisym="_delimiter_",
                           clfsym="_classify_",
                           unksym="<unk>")
    n > 12 && error("pretrain maximum layer: 12")
    #set_tokenizer((x)->nltk_word_tokenize(text_standardize(x)))
    emp = JSON.parsefile(dir("examples/gpt/pretrain/encoder_bpe_40000.json"))
    vocab = map(first, sort!(collect(emp), by=(x)->x.second))
    push!(vocab, startsym)
    push!(vocab, delisym)
    push!(vocab, clfsym)
    if unksym ∉ vocab
         pushfirst!(vocab,unk)
    end
    bpe = Bpe(joinpath(dir("examples/gpt/pretrain/vocab_40000.bpe")))
    vocab = IndexedDict(vocab)
    embed = Embed(output=768, input=length(vocab))
    pe = PositionEmbedding(768, 512; trainable=true)
    ce = CompositeEmbedding(embed, pe, unksym)
    gpt = Gpt(768, 12, 768*4, 12; act=GeLU(), pdrop=0.1)
    pms = load_gpt_pretrain_params()
    loadparams!(embed, [hcat(pms[2],randn(Float32,768, 3) .* 0.02)])
    loadparams!(pe, [pms[1]])
    for i = 1:n
        mhW = pms[12(i-1) + 3]
        mhb = pms[12(i-1) + 4]
        loadparams!(gpt.ts[i].mh.iqproj,[selectdim(mhW, 1, 1:768),
                                         selectdim(mhb, 1, 1:768)])
        loadparams!(gpt.ts[i].mh.ikproj,[selectdim(mhW, 1, 768+1:2*768),
                                         selectdim(mhb, 1, 768+1:2*768)])
        loadparams!(gpt.ts[i].mh.ivproj,[selectdim(mhW, 1, 2*768+1:3*768),
                                         selectdim(mhb, 1, 2*768+1:3*768)])
        loadparams!(gpt.ts[i].mh.oproj,[pms[12(i-1) + 5],
                                        pms[12(i-1) + 6]])
        loadparams!(gpt.ts[i].mhn,[pms[12(i-1) + 7],
                                   pms[12(i-1) + 8]])
        loadparams!(gpt.ts[i].pw.layers[1],[pms[12(i-1) + 9],
                                      pms[12(i-1) + 10]])
        loadparams!(gpt.ts[i].pw.layers[2],[pms[12(i-1) + 11],
                                       pms[12(i-1) + 12]])
        loadparams!(gpt.ts[i].pwn,[pms[12(i-1) + 13],
                                   pms[12(i-1) + 14]])
    end
    #gpt, embed, bpe, vocab
    TransformerModel(ce, gpt), bpe, vocab, gpt_tokenizer
end

struct CompositeEmbedding <: Layer
    tok::Embed
    pos::PositionEmbedding
    unksym::String
end

(m::CompositeEmbedding)(x) =  m.pos(m.tok(x))
   

function loadparams!(m::Layer, ws)
    for (w,wo) in zip(params(m),ws)
        copyto!(w, convert(arrtype,wo))
    end
end

"""
    getmask(ls::Container{<:Container})
get the mask for batched data.
"""
function getmask(ls::Container{<:Container})
    lens = map(length, ls)
    m = zeros(Int, maximum(lens), length(lens))

    for (i, l) ∈ enumerate(ls)
        m[1:length(l),i] .= 1
        #selectdim(selectdim(m, 2, i), 1, 1:length(l)) .= 1
    end
    m
end


function text_standardize(text)
    text = lowercase(text)
    text = replace(text, r"([a-z])(,|\.)"=>s"\1 \2")
    text = replace(text, "—"=>"-")
    text = replace(text, "–"=>"-")
    text = replace(text, "―"=>"-")
    text = replace(text, "…"=>"...")
    text = replace(text, "´"=>"'")
    text = replace(text, r"""(-+|~+|!+|"+|;+|\?+|\++|,+|\)+|\(+|\\+|\/+|\*+|\[+|\]+|}+|{+|\|+|_+)"""=>s" \1 ")
    text = replace(text, r"\s*\n\s*"=>" \n ")
    text = replace(text, r"[^\S\n]+"=>" ")
    strip(text)
end

end
