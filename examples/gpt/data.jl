using KnetLayers, ArgParse, BytePairEncoding
import KnetLayers: IndexedDict, arrtype, PadSequenceArray
using KnetLayers.Datasets
using KnetLayers.Datasets: StoryCloze


include("gpt.jl")

if isempty(ARGS)
    push!(ARGS,"rocstories")
end

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--gpu", "-g"
            help = "use gpu"
            action = :store_true
        "--epoch", "-e"
            help = "epoch"
            arg_type = Int
            default = 3
        "task"
            help = "task name"
            required = true
            range_tester = x -> x ∈ ["rocstories"]
    end

    return parse_args(ARGS, s)
end

const args = parse_commandline()

if args["gpu"]
    KnetLayers.settype!(KnetArray)
end


function transform(s1, s2, s3, s4, c1, c2, y)
    x = [startsym;
         segment(bpe, s1);
         segment(bpe, s2);
         segment(bpe, s3);
         segment(bpe, s4);
         delisym]
    x1 = [x; segment(bpe, c1); clfsym]
    x2 = [x; segment(bpe, c2); clfsym]

    x1, x2, y
end


function preprocess(batch, vocab_in, vocab_out)
    tdb = transform.(batch...)
    b1, b2, y = Datasets.batched(tdb)
    b1_mask = GPT.getmask(b1)
    b2_mask = GPT.getmask(b2)
    c1i = length.(b1)
    c2i = length.(b2)
    #c1i = [(findfirst(isequal(clfsym), x), i) for (i, x) in enumerate(b1)]
    #c2i = [(findfirst(isequal(clfsym), x), i) for (i, x) in enumerate(b2)]
    b1, b2 = map(b->vocab_in[b], b1), map(b->vocab_in[b], b2)
    #y = onehotbatch(y, labels)
    return PadSequenceArray(b1, pad=1)', PadSequenceArray(b2, pad=1)', c1i, c2i, vocab_out[y], b1_mask, b2_mask
end
