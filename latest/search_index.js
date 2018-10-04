var documenterSearchIndex = {"docs": [

{
    "location": "index.html#",
    "page": "Home",
    "title": "Home",
    "category": "page",
    "text": ""
},

{
    "location": "index.html#Welcome-to-KnetLayers.jl\'s-documentation!-1",
    "page": "Home",
    "title": "Welcome to KnetLayers.jl\'s documentation!",
    "category": "section",
    "text": ""
},

{
    "location": "index.html#KnetLayers-1",
    "page": "Home",
    "title": "KnetLayers",
    "category": "section",
    "text": "KnetLayers provides configurable deep learning layers for Knet, fostering your model development. You can use Knet and AutoGrad functionalities without adding them to current workspace."
},

{
    "location": "index.html#Example-Usages-1",
    "page": "Home",
    "title": "Example Usages",
    "category": "section",
    "text": "using KnetLayers\n#Instantiate an MLP model with random parameters\nmlp = MLP(100,50,20) # input size=100, hidden=50 and output=20\n#Do a prediction\nprediction = mlp(randn(Float32,100,1);activation=sigm) #defaul activation is relu\n\n#Instantiate Conv layer with random parameters\ncnn = Conv(3,3,3,10;padding=1,stride=1) # A conv filter with H=3,W=3,C=3,O=10\n#Filter your input\noutput = cnn(randn(Float32,224,224,3,1))\n\n#Instantiate an LSTM model\nlstm = LSTM(100,100;embed=50) #input size=100, hidden=100, embedding=50\n#You can use integers to represent one hot vectors\n#For example a pass over 5-Length sequence\ny,h,c,_ = lstm([3,2,1,4,5];hy=true,cy=true)\n#You can also use normal array inputs for low-level control\n#One iteration of LSTM\ny,h,c,_ = lstm(randn(100,1);hy=true,cy=true)\n#Pass over a 10-length sequence:\ny,h,c,_ = lstm(randn(100,1,10);hy=true,cy=true)\n#Pass over a mini-batch data which includes unequal length sequences\ny,h,c,_ = lstm([1,2,3,4],[5,6];sorted=true,hy=true,cy=true)\n#To see and modify rnn params in a structured view\nlstm.gatesview\n"
},

{
    "location": "index.html#Manual-1",
    "page": "Home",
    "title": "Manual",
    "category": "section",
    "text": "Pages = [\n \"reference.md\",\n]"
},

{
    "location": "reference.html#",
    "page": "Reference",
    "title": "Reference",
    "category": "page",
    "text": ""
},

{
    "location": "reference.html#Reference-1",
    "page": "Reference",
    "title": "Reference",
    "category": "section",
    "text": "ContentsPages = [\"reference.md\"]"
},

{
    "location": "reference.html#KnetLayers.Projection",
    "page": "Reference",
    "title": "KnetLayers.Projection",
    "category": "type",
    "text": "Projection(inputSize,projectSize;winit=xavier)\n\nCreates a projection layer according to given inputSize and projectSize.\n\n(m::Projection)(x) = m.w*x\n\nBy default parameters initialized with xavier, you can change it with winit argument\n\n\n\n\n\n"
},

{
    "location": "reference.html#KnetLayers.Embed",
    "page": "Reference",
    "title": "KnetLayers.Embed",
    "category": "type",
    "text": "Embed(inputSize,embedSize;winit=xavier)\n\nCreates an embedding layer according to given inputSize and embedSize.\n\nBy default embedding parameters initialized with xavier, you can change it winit argument\n\n(m::Embed)(x::Array{T}) where T<:Integer\n(m::Embed)(x; keepsize=true)\n\nEmbed objects are callable with an input which is either and integer array (one hot encoding) or an N-dimensional matrix. For N-dimensional matrix, size(x,1)==inputSize\n\n\n\n\n\n"
},

{
    "location": "reference.html#KnetLayers.Linear",
    "page": "Reference",
    "title": "KnetLayers.Linear",
    "category": "type",
    "text": "Linear(inputSize,outputSize;kwargs...)\n(m::Linear)(x; keepsize=true) #forward run\n\nCreates and linear layer according to given inputSize and outputSize.\n\nBy default embedding parameters initialized with xavier, you can change it winit argument\n\nKeywords\n\nwinit=xavier: weight initialization distribution\nbias=zeros: bias initialization distribution\n\n\n\n\n\n"
},

{
    "location": "reference.html#KnetLayers.Dense",
    "page": "Reference",
    "title": "KnetLayers.Dense",
    "category": "type",
    "text": "Dense(inputSize,outputSize;kwargs...)\n(m::Dense)(x; keepsize=true) #forward run\n\nCreates and deense layer according to given inputSize and outputSize.\n\nKeywords\n\nwinit=xavier: weight initialization distribution\nbias=zeros: bias initialization distribution\nf=ReLU(): activation function\nkeepsize=true: if false ndims(y)=2 all dimensions other than first one\n\nsqueezed to second dimension\n\n\n\n\n\n"
},

{
    "location": "reference.html#KnetLayers.BatchNorm",
    "page": "Reference",
    "title": "KnetLayers.BatchNorm",
    "category": "type",
    "text": "BatchNorm(channels:Int;options...)\n(m::BatchNorm)(x;training=false) #forward run\n\nOptions\n\nmomentum=0.1: A real number between 0 and 1 to be used as the scale of\n\nlast mean and variance. The existing running mean or variance is multiplied by  (1-momentum).\n\n`mean=nothing\': The running mean.\nvar=nothing: The running variance.\nmeaninit=zeros: The function used for initialize the running mean. Should either be nothing or\n\nof the form (eltype, dims...)->data. zeros is a good option.\n\nvarinit=ones: The function used for initialize the runn\n\nKeywords\n\ntraining=nothing: When training is true, the mean and variance of x are used and moments\n\nargument is modified if it is provided. When training is false, mean and variance  stored in the moments argument are used. Default value is true when at least one  of x and params is AutoGrad.Value, false otherwise.\n\n\n\n\n\n"
},

{
    "location": "reference.html#Core-Layers-1",
    "page": "Reference",
    "title": "Core Layers",
    "category": "section",
    "text": "KnetLayers.Projection\nKnetLayers.Embed   \nKnetLayers.Linear   \nKnetLayers.Dense   \nKnetLayers.BatchNorm   "
},

{
    "location": "reference.html#KnetLayers.ReLU",
    "page": "Reference",
    "title": "KnetLayers.ReLU",
    "category": "type",
    "text": "ReLU()\n(l::ReLU)(x) = max(0,x)\n\nRectified Linear Unit function.\n\n\n\n\n\n"
},

{
    "location": "reference.html#KnetLayers.Sigm",
    "page": "Reference",
    "title": "KnetLayers.Sigm",
    "category": "type",
    "text": "Sigm()\n(l::Sigm)(x) = sigm(x)\n\nSigmoid function\n\n\n\n\n\n"
},

{
    "location": "reference.html#KnetLayers.Tanh",
    "page": "Reference",
    "title": "KnetLayers.Tanh",
    "category": "type",
    "text": "Tanh()\n(l::Tanh)(x) = tanh(x)\n\nTangent hyperbolic function\n\n\n\n\n\n"
},

{
    "location": "reference.html#KnetLayers.ELU",
    "page": "Reference",
    "title": "KnetLayers.ELU",
    "category": "type",
    "text": "ELU()\n(l::ELU)(x) = elu(x) -> Computes x < 0 ? exp(x) - 1 : x\n\nExponential Linear Unit nonlineariy.\n\n\n\n\n\n"
},

{
    "location": "reference.html#KnetLayers.LeakyReLU",
    "page": "Reference",
    "title": "KnetLayers.LeakyReLU",
    "category": "type",
    "text": "LeakyReLU(α=0.2)\n(l::LeakyReLU)(x) -> Computes x < 0 ? α*x : x\n\n\n\n\n\n"
},

{
    "location": "reference.html#KnetLayers.Dropout",
    "page": "Reference",
    "title": "KnetLayers.Dropout",
    "category": "type",
    "text": "Dropout(p=0)\n\nDropout Layer. p is the droput probability.\n\n\n\n\n\n"
},

{
    "location": "reference.html#KnetLayers.SoftMax",
    "page": "Reference",
    "title": "KnetLayers.SoftMax",
    "category": "type",
    "text": "SoftMax(dims=:)\n(l::SoftMax)(x)\n\nTreat entries in x as as unnormalized scores and return softmax probabilities.\n\ndims is an optional argument, if not specified the normalization is over the whole x, otherwise the normalization is performed over the given dimensions. In particular, if x is a matrix, dims=1 normalizes columns of x and dims=2 normalizes rows of x.\n\n\n\n\n\n"
},

{
    "location": "reference.html#KnetLayers.LogSoftMax",
    "page": "Reference",
    "title": "KnetLayers.LogSoftMax",
    "category": "type",
    "text": "LogSoftMax(dims=:)\n(l::LogSoftMax)(x)\n\nTreat entries in x as as unnormalized log probabilities and return normalized log probabilities.\n\ndims is an optional argument, if not specified the normalization is over the whole x, otherwise the normalization is performed over the given dimensions. In particular, if x is a matrix, dims=1 normalizes columns of x and dims=2 normalizes rows of x.\n\n\n\n\n\n"
},

{
    "location": "reference.html#KnetLayers.LogSumExp",
    "page": "Reference",
    "title": "KnetLayers.LogSumExp",
    "category": "type",
    "text": "LogSumExp(dims=:)\n(l::LogSumExp)(x)\n\nCompute log(sum(exp(x);dims)) in a numerically stable manner.\n\ndims is an optional argument, if not specified the summation is over the whole x, otherwise the summation is performed over the given dimensions. In particular if x   is a matrix, dims=1 sums columns of x and dims=2 sums rows of x.\n\n\n\n\n\n"
},

{
    "location": "reference.html#Nonlinearities-1",
    "page": "Reference",
    "title": "Nonlinearities",
    "category": "section",
    "text": "KnetLayers.ReLU   \nKnetLayers.Sigm   \nKnetLayers.Tanh   \nKnetLayers.ELU   \nKnetLayers.LeakyReLU   \nKnetLayers.Dropout   \nKnetLayers.SoftMax   \nKnetLayers.LogSoftMax   \nKnetLayers.LogSumExp   "
},

{
    "location": "reference.html#KnetLayers.CrossEntropyLoss",
    "page": "Reference",
    "title": "KnetLayers.CrossEntropyLoss",
    "category": "type",
    "text": "CrossEntropyLoss(dims=1)\n(l::CrossEntropyLoss)(scores, answers::Array{<:Integer})\n\nCalculates negative log likelihood error on your predicted scores. answers should be integers corresponding to correct class indices. If an answer is 0, loss from that answer will not be included. This is usefull feature when you are working with unequal length sequences.\n\nif dims==1\n\nsize(scores) = C,[B,T1,T2,...]\nsize(answers)= [B,T1,T2,...]\n\nelseif dims==2\n\nsize(scores) = [B,T1,T2,...],C\nsize(answers)= [B,T1,T2,...]\n\n\n\n\n\n"
},

{
    "location": "reference.html#Loss-Functions-1",
    "page": "Reference",
    "title": "Loss Functions",
    "category": "section",
    "text": "KnetLayers.CrossEntropyLoss"
},

{
    "location": "reference.html#KnetLayers.Conv",
    "page": "Reference",
    "title": "KnetLayers.Conv",
    "category": "function",
    "text": "Conv(h,[w,c,o];kwargs...)\n\nCreates and convolutional layer according to given filter dimensions.\n\n(m::GenericConv)(x) #forward run\n\nIf m.w has dimensions (W1,W2,...,I,O) and x has dimensions (X1,X2,...,I,N), the result y will have dimensions (Y1,Y2,...,O,N) where\n\nYi=1+floor((Xi+2*padding[i]-Wi)/stride[i])\n\nHere I is the number of input channels, O is the number of output channels, N is the number of instances, and Wi,Xi,Yi are spatial dimensions. padding and stride are keyword arguments that can be specified as a single number (in which case they apply to all dimensions), or an tuple with entries for each spatial dimension.\n\nKeywords\n\nf=identity: nonlinear function applied after convolution\npool=nothing: Pooling layer or window size of pooling\nwinit=xavier: weight initialization distribution\nbias=zeros: bias initialization distribution\npadding=0: the number of extra zeros implicitly concatenated at the start and at the end of each dimension.\nstride=1: the number of elements to slide to reach the next filtering window.\nupscale=1: upscale factor for each dimension.\nmode=0: 0 for convolution and 1 for cross-correlation.\nalpha=1: can be used to scale the result.\nhandle: handle to a previously created cuDNN context. Defaults to a Knet allocated handle.\n\n\n\n\n\n"
},

{
    "location": "reference.html#KnetLayers.DeConv",
    "page": "Reference",
    "title": "KnetLayers.DeConv",
    "category": "function",
    "text": "DeConv(h,[w,c,o];kwargs...)\n\nCreates and deconvolutional layer according to given filter dimensions.\n\n(m::GenericConv)(x) #forward run\n\nIf m.w has dimensions (W1,W2,...,I,O) and x has dimensions (X1,X2,...,I,N), the result y will have dimensions (Y1,Y2,...,O,N) where\n\nYi = Wi+stride[i](Xi-1)-2padding[i]\n\nHere I is the number of input channels, O is the number of output channels, N is the number of instances, and Wi,Xi,Yi are spatial dimensions. padding and stride are keyword arguments that can be specified as a single number (in which case they apply to all dimensions), or an tuple with entries for each spatial dimension.\n\nKeywords\n\nf=identity: nonlinear function applied after convolution\nunpool=nothing: Unpooling layer or window size of unpooling\nwinit=xavier: weight initialization distribution\nbias=zeros: bias initialization distribution\npadding=0: the nßumber of extra zeros implicitly concatenated at the start and at the end of each dimension.\nstride=1: the number of elements to slide to reach the next filtering window.\nupscale=1: upscale factor for each dimension.\nmode=0: 0 for convolution and 1 for cross-correlation.\nalpha=1: can be used to scale the result.\nhandle: handle to a previously created cuDNN context. Defaults to a Knet allocated handle.\n\n\n\n\n\n"
},

{
    "location": "reference.html#KnetLayers.Pool",
    "page": "Reference",
    "title": "KnetLayers.Pool",
    "category": "function",
    "text": "Pool(kwargs...)\n(::GenericPool)(x)\n\nCompute pooling of input values (i.e., the maximum or average of several adjacent values) to produce an output with smaller height and/or width.\n\nCurrently 4 or 5 dimensional KnetArrays with Float32 or Float64 entries are supported. If x has dimensions (X1,X2,...,I,N), the result y will have dimensions (Y1,Y2,...,I,N) where\n\nYi=1+floor((Xi+2*padding[i]-window[i])/stride[i])\n\nHere I is the number of input channels, N is the number of instances, and Xi,Yi are spatial dimensions. window, padding and stride are keyword arguments that can be specified as a single number (in which case they apply to all dimensions), or an array/tuple with entries for each spatial dimension.\n\nKeywords:\n\nwindow=2: the pooling window size for each dimension.\npadding=0: the number of extra zeros implicitly concatenated at the\n\nstart and at the end of each dimension.\n\nstride=window: the number of elements to slide to reach the next pooling\n\nwindow.\n\nmode=0: 0 for max, 1 for average including padded values, 2 for average\n\nexcluding padded values.\n\nmaxpoolingNanOpt=0: Nan numbers are not propagated if 0, they are\n\npropagated if 1.\n\nalpha=1: can be used to scale the result.\n\n\n\n\n\n"
},

{
    "location": "reference.html#KnetLayers.UnPool",
    "page": "Reference",
    "title": "KnetLayers.UnPool",
    "category": "function",
    "text": "UnPool(kwargs...)\n(::GenericPool)(x)\n\nReverse of pooling. It has same kwargs with Pool\n\nx == pool(unpool(x;o...); o...)\n\n\n\n\n\n"
},

{
    "location": "reference.html#Convolutional-Layers-1",
    "page": "Reference",
    "title": "Convolutional Layers",
    "category": "section",
    "text": "KnetLayers.Conv   \nKnetLayers.DeConv   \nKnetLayers.Pool   \nKnetLayers.UnPool  "
},

{
    "location": "reference.html#KnetLayers.AbstractRNN",
    "page": "Reference",
    "title": "KnetLayers.AbstractRNN",
    "category": "type",
    "text": "SRNN(inputSize,hiddenSize;activation=:relu,options...)\nLSTM(inputSize,hiddenSize;options...)\nGRU(inputSize,hiddenSize;options...)\n\n(1) (l::T)(x;kwargs...) where T<:AbstractRNN\n(2) (l::T)(x::Array{Int};batchSizes=nothing,kwargs...) where T<:AbstractRNN\n(3) (l::T)(x::Vector{Vector{Int}};sorted=false,kwargs...) where T<:AbstractRNN\n\nAll RNN layers has above forward run(1,2,3) functionalities.\n\n(1) x is an input array with size equals d,[B,T]\n\n(2) For this You need to have an RNN with embedding layer. x is an integer array and inputs coressponds one hot vector indices. You can give 2D array for minibatching as rows corresponds to one instance. You can give 1D array with minibatching by specifying batch batchSizes argument. Checkout Knet.rnnforw for this.\n\n(3) For this You need to have an RNN with embedding layer. x is an vector of integer vectors. Every integer vector corresponds to an instance. It automatically batches inputs. It is better to give inputs as sorted. If your inputs sorted you can make sorted argument true to increase performance.\n\nsee RNNOutput\n\noptions\n\nembed=nothing: embedding size or and embedding layer\nnumLayers=1: Number of RNN layers.\nbidirectional=false: Create a bidirectional RNN if true.\ndropout=0: Dropout probability. Ignored if numLayers==1.\nskipInput=false: Do not multiply the input with a matrix if true.\ndataType=Float32: Data type to use for weights.\nalgo=0: Algorithm to use, see CUDNN docs for details.\nseed=0: Random number seed for dropout. Uses time() if 0.\nwinit=xavier: Weight initialization method for matrices.\nbinit=zeros: Weight initialization method for bias vectors.\nusegpu=(gpu()>=0): GPU used by default if one exists.\n\nKeywords\n\nhx=nothing : initial hidden states\ncx=nothing : initial memory cells\nhy=false   : if true returns h\ncy=false   : if true returns c\n\n\n\n\n\n"
},

{
    "location": "reference.html#KnetLayers.RNNOutput",
    "page": "Reference",
    "title": "KnetLayers.RNNOutput",
    "category": "type",
    "text": "struct RNNOutput\n    y\n    hidden\n    memory\n    indices\nend\n\nOutputs of the RNN models are always RNNOutput hidden,memory and indices may be nothing depending on the kwargs you used in forward.\n\ny is last hidden states of each layer. size(y)=(H/2H,[B,T]). If you use unequal length instances in a batch input, y becomes 2D array size(y)=(H/2H,sum_of_sequence_lengths). See indices and PadRNNOutput to get correct time outputs for a specific instance or to pad whole output.\n\nh is the hidden states in each timesstep. size(h) = h,B,L/2L\n\nc is the hidden states in each timesstep. size(h) = h,B,L/2L\n\nindices is corresponding instace indices for your RNNOutput.y. You may call yi = y[:,indices[i]].\n\n\n\n\n\n"
},

{
    "location": "reference.html#KnetLayers.PadSequenceArray",
    "page": "Reference",
    "title": "KnetLayers.PadSequenceArray",
    "category": "function",
    "text": "PadSequenceArray(batch::Vector{Vector{T}}) where T<:Integer\n\nPads a batch of integer arrays with zeros\n\njulia> PadSequenceArray([[1,2,3],[1,2],[1]]) 3×3 Array{Int64,2}:  1  2  3  1  2  0  1  0  0\n\n\n\n\n\n"
},

{
    "location": "reference.html#KnetLayers.PadRNNOutput",
    "page": "Reference",
    "title": "KnetLayers.PadRNNOutput",
    "category": "function",
    "text": "PadRNNOutput(s::RNNOutput)\n\nPads a rnn output if it is produces by unequal length batches size(s.y)=(H/2H,sum_of_sequence_lengths) becomes (H/2H,B,Tmax)\n\n\n\n\n\n"
},

{
    "location": "reference.html#Recurrent-Layers-1",
    "page": "Reference",
    "title": "Recurrent Layers",
    "category": "section",
    "text": "KnetLayers.AbstractRNN  \nKnetLayers.RNNOutput\nKnetLayers.PadSequenceArray\nKnetLayers.PadRNNOutput"
},

{
    "location": "reference.html#KnetLayers.MLP",
    "page": "Reference",
    "title": "KnetLayers.MLP",
    "category": "type",
    "text": "MLP(h::Int...;kwargs...)\n\nCreates a multi layer perceptron according to given hidden states. First hidden state is equal to input size and the last one equal to output size.\n\n(m::MLP)(x;prob=0)\n\nRuns MLP with given input x. prob is the dropout probability.\n\nKeywords\n\nwinit=xavier: weight initialization distribution\nbias=zeros: bias initialization distribution\nf=ReLU(): activation function\n\n\n\n\n\n"
},

{
    "location": "reference.html#Special-Layers-1",
    "page": "Reference",
    "title": "Special Layers",
    "category": "section",
    "text": "KnetLayers.MLP   "
},

{
    "location": "reference.html#Function-Index-1",
    "page": "Reference",
    "title": "Function Index",
    "category": "section",
    "text": "Pages = [\"reference.md\"]"
},

]}