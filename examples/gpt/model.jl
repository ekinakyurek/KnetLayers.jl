using KnetLayers, BytePairEncoding
import KnetLayers: IndexedDict

ENV["DATADEPS_ALWAYS_ACCEPT"] = true

const startsym = "_start_"
const delisym = "_deli_"
const clfsym = "_clf_"
const unksym = "<unk>"
const labels = ["1", "2"]

const vocab_out = IndexedDict(labels)

gpt_model, bpe, vocab_in, tokenizer = GPT.load_gpt_pretrain(12,
                                                    startsym=startsym,
                                                    delisym=delisym,
                                                    clfsym=clfsym,
                                                    unksym=unksym)


set_tokenizer(tokenizer)

gpt = Chain(gpt_model, Dense(input=768,output=1,activation=nothing), Dropout(0.1), CrossEntropyLoss())
embed = gpt[1].embed.tok


function loss(gpt, x1, x2, y, x1_mask, x2_mask, c1_index, c2_index)
    e1 = gpt[1].embed(x1)
    e2 = gpt[1].embed(x2)

    t1 = gpt[1].transformers(e1, x1_mask)
    t2 = gpt[1].transformers(e2, x2_mask)
    lm = GPT.lmloss(embed, x1, t1, x1_mask) + GPT.lmloss(embed, x2, t2, x2_mask)

    c1 = hcat((t1[:,v,i] for (i,v) in enumerate(c1_index))...)
    c2 = hcat((t2[:,v,i] for (i,v) in enumerate(c2_index))...)
    p1 = gpt[2](gpt[3](c1))
    p2 = gpt[2](gpt[3](c2))
    p = vcat(p1, p2)

    cl = gpt[4](p, y)
    #unstable type will cause performance issue
    0.5f0 * lm + cl
end

function predict(gpt, x1, x2, y, x1_mask, x2_mask, c1_index, c2_index)
    e1 = gpt[1].embed(x1)
    e2 = gpt[1].embed(x2)
    t1 = gpt[1].transformers(e1, x1_mask)
    t2 = gpt[1].transformers(e2, x2_mask)
    lm = GPT.lmloss(embed, x1, t1, x1_mask) + GPT.lmloss(embed, x2, t2, x2_mask)
    c1 = hcat((t1[:,v,i] for (i,v) in enumerate(c1_index))...)
    c2 = hcat((t2[:,v,i] for (i,v) in enumerate(c2_index))...)
    p1 = gpt[2](gpt[3](c1))
    p2 = gpt[2](gpt[3](c2))
    p  = vcat(p1, p2)
    cl = gpt[4](p, y)
    #unstable type will cause performance issue
    p
end

const rocs = Datasets.StoryCloze()
const ps = params(gpt)
const opt = Adam(lr=6.25e-5)
setoptim!(M, optimizer) = for p in params(M); p.opt = deepcopy(optimizer); end
lrdecay!(M, decay::Real) = for p in params(M); p.opt.lr = p.opt.lr*decay; end



function eval(gpt,data,vocab_in, vocab_out; Batch=8)
    println("eval:")
    datas = dataset(Datasets.Test, data)
    al,i = 0.0, 0
    while (batch = get_batch(datas, Batch)) !== nothing
        b1, b2, c1i, c2i, y, b1_mask, b2_mask = preprocess(batch, vocab_in, vocab_out)
        p = predict(gpt, b1, b2, y, b1_mask, b2_mask, c1i, c2i)
        a = accuracy(p, y)
        al += a
        i  += 1
    end
    total =  al /= i
    @show total
    return total
end

function train1!(gpt, data, vocab_in, vocab_out; opt=nothing, epoch=10, Batch=8)
    #global Batch, rocs, opt, ps
    ps  = params(gpt)
    if opt !== Nothing
        setoptim!(gpt,opt)
    end
    lss = typemax(Float64)
    for e = 1:epoch
        println("start training: $e")
        datas  = dataset(Datasets.Train, data)
        i, al  = 0, 0.0
        grads = nothing

        while (batch = get_batch(datas,Batch)) !== nothing
            b1, b2, c1i, c2i, y, b1_mask, b2_mask, = preprocess(batch, vocab_in, vocab_out)
            J =  @diff loss(gpt, b1, b2, y, b1_mask, b2_mask, c1i, c2i)
            if grads == nothing
                grads = map(w->grad(J,w),ps)
            else
                for (k,w) in enumerate(ps)
                    grads[k] += grad(J,w)
                end
            end
            
            if i % 4 == 0 
                for (k,w) in enumerate(ps)
                    update!(w.value,grads[k]/4.0f0,w.opt)
                end
                for g in grads
                    fill!(g,0.0f0)
                end
            end
            
            i  += 1
            al += value(J)
            if i%100==1
                println(al/i)
            end
        end
        nwlss = al/i
        if nwlss < lss
            lss=nwlss
        else
            lrdecay!(gpt,0.002)
            println("lr decay!")
        end
        total = eval(gpt, data, vocab_in, vocab_out)
    end
end

# function accuracy(p, y)
#     pred = onecold(collect(p))
#     label = onecold(collect(y))
#     sum(pred .== label) / length(label)
# end

#
#
# const embed = gpt.embed.embeddings.tok
# const clf = arrtype(Dense(input=768, output=1))
# const ansdrop = Dropout(0.1)
# const loss = CrossEntropyLoss()
#
#
# function loss(x1, x2, y, x1_mask, x2_mask, c1_index, c2_index; train=false)
#     e1 = gpt.embed(x1)
#     e2 = gpt.embed(x2)
#     t1 = gpt.transformers(e1, x1_mask)
#     t2 = gpt.transformers(e2, x2_mask)
#     lm = lmloss(embed, vocab[x1], t1, x1_mask) + lmloss(embed, vocab[x2], t2, x2_mask)
#     c1 = t1[:,c1_index]
#     c2 = t2[:,c2_index]
#
#     p1 = clf(c1)
#     p2 = clf(c2)
#     p = vcat(p1, p2)
#     p = ansdrop(p, 1)
#     cl = loss(p, y)
#     if train
#         convert(eltype(arrtype), 0.5) * lm
#     else
#         convert(eltype(arrtype), 0.5) * lm + cl,p
#     end
# end
#
# const rocs = StoryCloze()
# setoptim!(params([gpt, clf]); lr=ADAM(6.25e-5))
#
# train!(args["epoch"])
