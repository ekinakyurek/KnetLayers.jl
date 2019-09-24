import AutoGrad: @primitive, @zerograd
"""
`batchedmul(A, B))` performs a batch matrix-matrix product of matrices stored in `A`
and `B`. `A` and `B` must be 3d and the last dimension represents the batch size.
If A is a (m,n,b) tensor, B is a (n,k,b) tensor, and the output is a (m,k,b) tensor.
"""
function batchedmul(A, B; transA::Bool = false, transB::Bool = false)
    T = eltype(A)
    (bs = size(A, 3)) == size(B, 3) || error("batch size mismatch")
    C = similar(A, size(A, transA ? 2 : 1), size(B, transB ? 1 : 2), bs)
    At = transA ? 'T' : 'N'
    Bt = transB ? 'T' : 'N'
    batched_gemm!(At, Bt, one(T), A, B, zero(T), C)
    return C
end

function batchedmul(A::KnetArray, B::KnetArray{T}) where {T}
    m,n,bs = size(A)
    n,k,bs = size(B)
    return batchedmul!('N','N',one(T),A,B,zero(T),similar(A, (m, k, bs)))
end

function batchedmul!(transA::AbstractChar, transB::AbstractChar, alpha::Number, A::KnetArray{T}, B::KnetArray{T}, beta::Number, C::KnetArray{T}) where {T}
    cublasop(c::Char)=(if c=='N'; 0; elseif c=='T'; 1; elseif c=='C'; 2; else error("Unknown cublas op $c"); end)
    if ndims(A) != 3 || ndims(B) != 3
        throw(DimensionMismatch("$(map(size,(A,B,C)))"))
    end
    ma,ka,bsa = size(A)
    kb,nb,bsb = size(B)
    if bsa != bsb
        throw(DimensionMismatch("$(map(size,(A,B,C)))"))
    end
    bs = bsa
    if transA == 'N'
        m=ma; k=ka;
    else
        m=ka; k=ma;
    end
    if transB == 'N'
        k == kb || throw(DimensionMismatch("$(map(size,(A,B,C)))"))
        n=nb; k==kb;
    else
        k == nb || throw(DimensionMismatch("$(map(size,(A,B,C)))"))
        n=kb; k==nb;
    end
    (m == size(C,1) && n == size(C,2) && bs == size(C,3)) || throw(DimensionMismatch("$(map(size,(A,B,C)))"))
    lda,ldb,ldc=ma,ka,ma
    transa = Knet.cublasop(transA); transb = cublasop(transB)
    alpha = T[alpha]; beta = T[beta]
    strideA, strideB, strideC = m*k, k*n, m*n
    if T<:Float64
        Knet.@cublas(cublasDgemmStridedBatched, (Cptr, UInt32, UInt32, Cint, Cint, Cint, Ptr{T}, Ptr{T}, Cint, Clonglong, Ptr{T}, Cint, Clonglong, Ptr{T}, Ptr{T}, Cint, Clonglong, Cint), cublashandle(), transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, bs)
    elseif T<:Float32
        Knet.@cublas(cublasSgemmStridedBatched, (Cptr, UInt32, UInt32, Cint, Cint, Cint, Ptr{T}, Ptr{T}, Cint, Clonglong, Ptr{T}, Cint, Clonglong, Ptr{T}, Ptr{T}, Cint, Clonglong, Cint), cublashandle(), transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, bs)
    else
        error("CUBLAS does not support $T")
    end
    return C
end

#batched cpu gemm by BatchedRoutines.jl
for (gemm, elty) in
    ((:dgemm_,:Float64),
     (:sgemm_,:Float32),)
    @eval begin
        function batched_gemm!(transA::AbstractChar,
                               transB::AbstractChar,
                               alpha::($elty),
                               A::AbstractArray{$elty, 3},
                               B::AbstractArray{$elty, 3},
                               beta::($elty),
                               C::AbstractArray{$elty, 3})
            @assert !LinearAlgebra.BLAS.has_offset_axes(A, B, C)
            @assert size(A, 3) == size(B, 3) == size(C, 3) "batch size mismatch"
            m = size(A, transA == 'N' ? 1 : 2)
            ka = size(A, transA == 'N' ? 2 : 1)
            kb = size(B, transB == 'N' ? 1 : 2)
            n = size(B, transB == 'N' ? 2 : 1)
            if ka != kb || m != size(C,1) || n != size(C,2)
                throw(DimensionMismatch("A has size ($m,$ka), B has size ($kb,$n), C has size $(size(C))"))
            end
            LinearAlgebra.BLAS.chkstride1(A)
            LinearAlgebra.BLAS.chkstride1(B)
            LinearAlgebra.BLAS.chkstride1(C)

            ptrA = Base.unsafe_convert(Ptr{$elty}, A)
            ptrB = Base.unsafe_convert(Ptr{$elty}, B)
            ptrC = Base.unsafe_convert(Ptr{$elty}, C)

            for k in 1:size(A, 3)
                ccall((LinearAlgebra.BLAS.@blasfunc($gemm), LinearAlgebra.BLAS.libblas), Cvoid,
                    (Ref{UInt8}, Ref{UInt8}, Ref{LinearAlgebra.BLAS.BlasInt}, Ref{LinearAlgebra.BLAS.BlasInt},
                     Ref{LinearAlgebra.BLAS.BlasInt}, Ref{$elty}, Ptr{$elty}, Ref{LinearAlgebra.BLAS.BlasInt},
                     Ptr{$elty}, Ref{LinearAlgebra.BLAS.BlasInt}, Ref{$elty}, Ptr{$elty},
                     Ref{LinearAlgebra.BLAS.BlasInt}),
                     transA, transB, m, n,
                     ka, alpha, ptrA, max(1,stride(A,2)),
                     ptrB, max(1,stride(B,2)), beta, ptrC,
                     max(1,stride(C,2)))

                ptrA += size(A, 1) * size(A, 2) * sizeof($elty)
                ptrB += size(B, 1) * size(B, 2) * sizeof($elty)
                ptrC += size(C, 1) * size(C, 2) * sizeof($elty)
            end

            C
        end
    end
end


@primitive batchedmul(x1,x2; transA::Bool=false, transB::Bool=false),dy,y (transA ? batchedmul(x2, dy; transA=transB , transB=true) :  batchedmul(dy, x2;  transA=false, transB=!transB) )    (transB ? batchedmul(dy,x1; transA=true , transB= !transA) :  batchedmul(x1, dy;  transA=!transA , transB=false))
@zerograd  batchedmul!(transA::AbstractChar, transB::AbstractChar, alpha::Number, A::KnetArray, B::KnetArray, beta::Number, C::KnetArray)
@zerograd  batchedmul!(A, B, C)
