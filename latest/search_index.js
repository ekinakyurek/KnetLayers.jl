var documenterSearchIndex = {"docs": [

{
    "location": "index.html#",
    "page": "Home",
    "title": "Home",
    "category": "page",
    "text": ""
},

{
    "location": "index.html#Welcome-to-KLayers.jl\'s-documentation!-1",
    "page": "Home",
    "title": "Welcome to KLayers.jl\'s documentation!",
    "category": "section",
    "text": ""
},

{
    "location": "index.html#KLayers-1",
    "page": "Home",
    "title": "KLayers",
    "category": "section",
    "text": "KLayers provides configurable deep learning layers for Knet, fostering your model development. You can use Knet and AutoGrad functionalities without adding them to current workspace."
},

{
    "location": "index.html#Example-Usages-1",
    "page": "Home",
    "title": "Example Usages",
    "category": "section",
    "text": "using KLayers\n#Instantiate an MLP model with random parameters\nmlp = MLP(100,50,20) # input size=100, hidden=50 and output=20\n#Do a prediction\nprediction = mlp(randn(Float32,100,1);activation=sigm) #defaul activation is relu\n\n#Instantiate Conv layer with random parameters\ncnn = Conv(3,3,3,10;padding=1,stride=1) # A conv filter with H=3,W=3,C=3,O=10\n#Filter your input\noutput = cnn(randn(Float32,224,224,3,1))\n\n#Instantiate an LSTM model\nlstm = LSTM(100,100;embed=50) #input size=100, hidden=100, embedding=50\n#You can use integers to represent one hot vectors\n#For example a pass over 5-Length sequence\ny,h,c,_ = lstm([3,2,1,4,5];hy=true,cy=true)\n#You can also use normal array inputs for low-level control\n#One iteration of LSTM\ny,h,c,_ = lstm(randn(100,1);hy=true,cy=true)\n#Pass over a 10-length sequence:\ny,h,c,_ = lstm(randn(100,1,10);hy=true,cy=true)\n#Pass over a mini-batch data which includes unequal length sequences\ny,h,c,_ = lstm([1,2,3,4],[5,6];sorted=true,hy=true,cy=true)\n#To see and modify rnn params in a structured view\nlstm.gatesview\n"
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
    "location": "reference.html#KLayers.Embed",
    "page": "Reference",
    "title": "KLayers.Embed",
    "category": "type",
    "text": "Embed(inputSize,embedSize;winit=xavier)\n\nCreates and embedding layer according to given inputSize and embedSize.\n\nBy default embedding parameters initialized with xavier, you can change it winit argument\n\n(m::Embed)(x::Array{T}) where T<:Integer\n(m::Embed)(x)\n\nEmbed objects are callable with an input which is either and integer array (one hot encoding) or an N-dimensional matrix. For N-dimensional matrix, size(x,1)==inputSize\n\n\n\n\n\n"
},

{
    "location": "reference.html#KLayers.Linear",
    "page": "Reference",
    "title": "KLayers.Linear",
    "category": "type",
    "text": "Linear(inputSize,outputSize;kwargs...)\n(m::Linear)(x) #forward run\n\nCreates and linear layer according to given inputSize and outputSize.\n\nBy default embedding parameters initialized with xavier, you can change it winit argument\n\nKeywords\n\nwinit=xavier: weight initialization distribution\nbias=zeros: bias initialization distribution\n\n\n\n\n\n"
},

{
    "location": "reference.html#KLayers.Dense",
    "page": "Reference",
    "title": "KLayers.Dense",
    "category": "type",
    "text": "Dense(inputSize,outputSize;kwargs...)\n(m::Dense)(x) #forward run\n\nCreates and deense layer according to given inputSize and outputSize.\n\nKeywords\n\nwinit=xavier: weight initialization distribution\nbias=zeros: bias initialization distribution\nf=ReLU(): activation function\n\n\n\n\n\n"
},

{
    "location": "reference.html#KLayers.Conv",
    "page": "Reference",
    "title": "KLayers.Conv",
    "category": "type",
    "text": "Conv(h,[w,c,o];kwargs...)\n(m::Conv)(x) #forward run\n\nCreates and convolutional layer according to given filter dimensions.\n\nKeywords\n\nwinit=xavier: weight initialization distribution\nbias=zeros: bias initialization distribution\npadding=0: the number of extra zeros implicitly concatenated at the start and at the end of each dimension.\nstride=1: the number of elements to slide to reach the next filtering window.\nupscale=1: upscale factor for each dimension.\nmode=0: 0 for convolution and 1 for cross-correlation.\nalpha=1: can be used to scale the result.\nhandle: handle to a previously created cuDNN context. Defaults to a Knet allocated handle.\n\n\n\n\n\n"
},

{
    "location": "reference.html#KLayers.BatchNorm",
    "page": "Reference",
    "title": "KLayers.BatchNorm",
    "category": "type",
    "text": "BatchNorm(channels:Int;o...)\n(m::BatchNorm)(x;o...) #forward run\n\n\n\n\n\n"
},

{
    "location": "reference.html#Core-1",
    "page": "Reference",
    "title": "Core",
    "category": "section",
    "text": "KLayers.Embed   \nKLayers.Linear   \nKLayers.Dense   \nKLayers.Conv   \nKLayers.BatchNorm   "
},

{
    "location": "reference.html#KLayers.ReLU",
    "page": "Reference",
    "title": "KLayers.ReLU",
    "category": "type",
    "text": "ReLU()\n(l::ReLU)(x) = max(0,x)\n\nFast kernel is avaiable for gpu\n\n\n\n\n\n"
},

{
    "location": "reference.html#KLayers.Sigm",
    "page": "Reference",
    "title": "KLayers.Sigm",
    "category": "type",
    "text": "Sigm()\n(l::Sigm)(x) = sigm(x)\n\nSigmoid function\n\n\n\n\n\n"
},

{
    "location": "reference.html#KLayers.Tanh",
    "page": "Reference",
    "title": "KLayers.Tanh",
    "category": "type",
    "text": "Tanh()\n(l::Tanh)(x) = tanh(x)\n\nTangent hyperbolic function\n\n\n\n\n\n"
},

{
    "location": "reference.html#KLayers.ELU",
    "page": "Reference",
    "title": "KLayers.ELU",
    "category": "type",
    "text": "ELU()\n(l::ELU)(x) = elu(x) -> Computes x < 0 ? exp(x) - 1 : x\n\n\n\n\n\n"
},

{
    "location": "reference.html#KLayers.LeakyReLU",
    "page": "Reference",
    "title": "KLayers.LeakyReLU",
    "category": "type",
    "text": "LeakyReLU(α=0.2)\n(l::ELU)(x) -> Computes x < 0 ? α*x : x\n\n\n\n\n\n"
},

{
    "location": "reference.html#KLayers.Dropout",
    "page": "Reference",
    "title": "KLayers.Dropout",
    "category": "type",
    "text": "Dropout(p=0)\n\nDropout Layer. p is the droput probability.\n\n\n\n\n\n"
},

{
    "location": "reference.html#KLayers.SoftMax",
    "page": "Reference",
    "title": "KLayers.SoftMax",
    "category": "type",
    "text": "SoftMax(dims=:)\n(l::SoftMax)(x)\n\nTreat entries in x as as unnormalized scores and return softmax probabilities.\n\ndims is an optional argument, if not specified the normalization is over the whole x, otherwise the normalization is performed over the given dimensions. In particular, if x is a matrix, dims=1 normalizes columns of x and dims=2 normalizes rows of x.\n\n\n\n\n\n"
},

{
    "location": "reference.html#KLayers.LogP",
    "page": "Reference",
    "title": "KLayers.LogP",
    "category": "type",
    "text": "LogP(dims=:)\n(l::LogP)(x)\n\nTreat entries in x as as unnormalized log probabilities and return normalized log probabilities.\n\ndims is an optional argument, if not specified the normalization is over the whole x, otherwise the normalization is performed over the given dimensions. In particular, if x is a matrix, dims=1 normalizes columns of x and dims=2 normalizes rows of x.\n\n\n\n\n\n"
},

{
    "location": "reference.html#KLayers.LogSumExp",
    "page": "Reference",
    "title": "KLayers.LogSumExp",
    "category": "type",
    "text": "LogSumExp(dims=:)\n(l::LogSumExp)(x)\n\nCompute log(sum(exp(x);dims)) in a numerically stable manner.\n\ndims is an optional argument, if not specified the summation is over the whole x, otherwise the summation is performed over the given dimensions. In particular if x   is a matrix, dims=1 sums columns of x and dims=2 sums rows of x.\n\n\n\n\n\n"
},

{
    "location": "reference.html#Nonlinear-1",
    "page": "Reference",
    "title": "Nonlinear",
    "category": "section",
    "text": "KLayers.ReLU   \nKLayers.Sigm   \nKLayers.Tanh   \nKLayers.ELU   \nKLayers.LeakyReLU   \nKLayers.Dropout   \nKLayers.SoftMax   \nKLayers.LogP   \nKLayers.LogSumExp   "
},

{
    "location": "reference.html#KLayers.MLP",
    "page": "Reference",
    "title": "KLayers.MLP",
    "category": "type",
    "text": "MLP(h::Int...;kwargs...)\n\nCreates a multi layer perceptron according to given hidden states. First hidden state is equal to input size and the last one equal to output size.\n\n(m::MLP)(x;prob=0)\n\nRuns MLP with given input x. prob is the dropout probability.\n\nKeywords\n\nwinit=xavier: weight initialization distribution\nbias=zeros: bias initialization distribution\nf=ReLU(): activation function\n\n\n\n\n\n"
},

{
    "location": "reference.html#MLP-1",
    "page": "Reference",
    "title": "MLP",
    "category": "section",
    "text": "KLayers.MLP   "
},

{
    "location": "reference.html#KLayers.RNN",
    "page": "Reference",
    "title": "KLayers.RNN",
    "category": "type",
    "text": "SRNN(inputSize,hiddenSize;activation=:relu,options...)\nLSTM(inputSize,hiddenSize;options...)\nGRU(inputSize,hiddenSize;options...)\n\n(1) (l::T)(x;kwargs...) where T<:RNN\n(2) (l::T)(x::Array{Int};batchSizes=nothing,kwargs...) where T<:RNN\n(3) (l::T)(x::Vector{Vector{Int}};sorted=false,kwargs...) where T<:RNN\n\nAll RNN layers has above forward run(1,2,3) functionalities.\n\n(1) x is an input array with size equals d,[B,T]\n\n(2) For this You need to have an RNN with embedding layer. x is an integer array and inputs coressponds one hot vector indices. You can give 2D array for minibatching as rows corresponds to one instance. You can give 1D array with minibatching by specifying batch batchSizes argument. Checkout Knet.rnnforw for this.\n\n(3) For this You need to have an RNN with embedding layer. x is an vector of integer vectors. Every integer vector corresponds to an instance. It automatically batches inputs. It is better to give inputs as sorted. If your inputs sorted you can make sorted argument true to increase performance.\n\nOutputs of the forward functions are always y,h,c,indices. h,c and indices may be nothing depending on the kwargs you used in forward.\n\ny is last hidden states of each layer. size(y)=(H/2H,[B,T]). If you use batchSizes argument y becomes 2D array size(y)=(H/2H,sum(batchSizes)). To get correct hidden states for an instance in your batch you should use indices output.\n\nh is the hidden states in each timesstep. size(h) = h,B,L/2L\n\nc is the hidden states in each timesstep. size(h) = h,B,L/2L\n\nindices is corresponding indices for your batches in y if you used batchSizes. To get ith instance\'s hidden states in each times step, you may type: y[:,indices[i]] `\n\noptions\n\nembed=nothing: embedding size or and embedding layer\nnumLayers=1: Number of RNN layers.\nbidirectional=false: Create a bidirectional RNN if true.\ndropout=0: Dropout probability. Ignored if numLayers==1.\nskipInput=false: Do not multiply the input with a matrix if true.\ndataType=Float32: Data type to use for weights.\nalgo=0: Algorithm to use, see CUDNN docs for details.\nseed=0: Random number seed for dropout. Uses time() if 0.\nwinit=xavier: Weight initialization method for matrices.\nbinit=zeros: Weight initialization method for bias vectors.\nusegpu=(gpu()>=0): GPU used by default if one exists.\n\nkwargs\n\nhx=nothing : initial hidden states\ncx=nothing : initial memory cells\nhy=false   : if true returns h\ncy=false   : if true returns c\n\n\n\n\n\n"
},

{
    "location": "reference.html#RNN-1",
    "page": "Reference",
    "title": "RNN",
    "category": "section",
    "text": "KLayers.RNN      "
},

{
    "location": "reference.html#Function-Index-1",
    "page": "Reference",
    "title": "Function Index",
    "category": "section",
    "text": "Pages = [\"reference.md\"]"
},

]}
