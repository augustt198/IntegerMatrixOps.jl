module IntegerMatrixOps

using LinearAlgebra
using Primes

export detsign, detbig, permbig


@inline fmod(x, m) = muladd(-m, round(x * inv(m)), x)
@inline modinv(x, m) = gcdx(x, m)[2]
@inline fmodinv(x, m) = convert(typeof(x), modinv(Int(x), m))

macro modops(m)
    esc(quote
        @inline +ₘ(x, y) = fmod(x + y, $m)
        @inline -ₘ(x, y) = fmod(x - y, $m)
        @inline *ₘ(x, y) = fmod(x * y, $m)
        @inline invₘ(x)  = fmodinv(x, $m)
        @inline /ₘ(x, y) = x *ₘ invₘ(y)
        @inline det2ₘ(a, b, c, d) = fmod(muladd(a, d, -b*c), $m)
    end)
end

function build_W(mods)::Matrix{Float64} # fix this shit
    k = length(mods)
    fn(i, j) = fmodinv(
        reduce((p, ℓ) -> fmod(p * ifelse(ℓ==i, 1, mods[ℓ]), mods[i]), 1:j; init=float(1)),
        mods[i])
    [fn(i, j) for i = 1:k, j = 1:k]
end

const PRECOMP_PRIMES = begin
    _max = Int64(sqrt(1/eps(Float64)))
    accumulate((p, _) -> prevprime(p - 1), 1:128, init=_max)
end

const LOG_SUM_PRECOMP_PRIMES = accumulate(+, log2.(PRECOMP_PRIMES))
const PRECOMP_W = build_W(PRECOMP_PRIMES)

function build_lagrange(Ps)
    M = accumulate(*, Ps, init=big(1))

    function V_ij(i, j)
        q, qmod = big(1), 1
        for ℓ = 1:j
            p = (ℓ == i) ? 1 : Ps[ℓ]
            Base.GMP.MPZ.mul_si!(q, p)
            qmod = mod(qmod * p, Ps[i])
        end
        return Base.GMP.MPZ.mul_si!(q, invmod(qmod, Ps[i]))
    end

    M, V_ij.(axes(Ps, 1), axes(Ps, 1)')
end

const PRECOMP_LAGRANGE = build_lagrange(PRECOMP_PRIMES)
lagrange(Ps) = (Ps === PRECOMP_PRIMES) ? PRECOMP_LAGRANGE : build_lagrange(Ps)

# Upper bounds for some functions of A
function bounds(A)
    N = size(A, 1)
    logsum = max2 = zero(float(eltype(A)))
    for col in eachcol(A)
        s = 0
        for x in col
            m2 = abs2(float(x))
            s += m2
            max2 = max(max2, m2)
        end
        logsum += log2(s)
    end

    L_A = ceil(log2(max2)/2) + 2
    L_det = N * (log2(N) + L_A)

    (;  :lg_hadamard => logsum/2,
        :lg_perm => 0.5*N*log2(N) + logsum/2,
        :lg_bareiss => L_det,
        :max => sqrt(max2))
end

function permanentmod(A, m)
    @modops(m)

    N = size(A, 1)
    outer_sum = zero(eltype(A))
    inner_sums = (similar(A, eltype(A), (N,)) .= 0)

    prev = 0
    @inbounds for idx = 1:(2^N - 1)
        gray = idx ⊻ (idx >> 1)
        diff = gray ⊻ prev
        prev = gray
        
        j = trailing_zeros(diff) + 1

        prod = one(outer_sum)
        for i = 1:N
            sgn = (-1) ^ iszero(gray & diff)
            inner_sums[i] = inner_sums[i] +ₘ (sgn * A[i, j])
            prod = prod *ₘ inner_sums[i]
        end

        sgn = (-1) ^ isodd(count_ones(gray))
        outer_sum = outer_sum +ₘ (sgn * prod)
    end

    return (-1) ^ N * outer_sum
end

# moduli needed for computing determinant
function modsfor(lg_bd)
    primes = PRECOMP_PRIMES
    logsum = 0
    n = 0
    while logsum < lg_bd
        n += 1
        if n <= length(PRECOMP_PRIMES)
            logsum = LOG_SUM_PRECOMP_PRIMES[n]
        else
            if n == length(PRECOMP_PRIMES) + 1
                primes = copy(PRECOMP_PRIMES)
            end
            p = prevprime(primes[end] - 1)
            logsum += log2(p)
            push!(primes, p)
        end
    end

    # todo: make the build_W case better
    if parent(primes) === PRECOMP_PRIMES
        W = PRECOMP_W
    else
        W = build_W(primes)
    end

    n, primes, W
end

function gauss_determinant!(A, m)
    @modops(m)

    N = size(A, 1)

    (N == 2) && return det2(A, m)
    (N == 3) && return det3(A, m)
    (N == 4) && return det4(A, m)

    partial = d = one(eltype(A))
    
    @inbounds for i = 1:(N - 4)
        piv = 0
        for k = i:N
            if !iszero(A[i, k])
                piv = k
                break
            end
        end
        # singular
        (piv == 0) && return zero(eltype(A))
        
        if piv != i
            d = 0 -ₘ d
            Base.swapcols!(A, piv, i)
        end
        
        a_ii = A[i, i]
        partial = partial *ₘ a_ii
        d = d *ₘ partial

        for j = (i+1):N
            a_ij = A[i, j]
            for k = (i+1):N
                A[k, j] = det2ₘ(-a_ij, -a_ii, A[k, j], A[k, i])
            end
        end
    end

    rest = det4((@view A[(N - 3):N, (N - 3):N]), m)
    return rest /ₘ ((partial *ₘ partial) *ₘ d) # return (@inbounds partial *ₘ A[N, N]) /ₘ d
end

function rank!(A, m)
    @modops(m)

    N, M = size(A)
    i = j = 1
    rank = N
    @inbounds while i <= M && j <= N
        pivot_indices = j:N
        pidx = findfirst(k -> !iszero(@inbounds A[i, k]), pivot_indices)
        if isnothing(pidx)
            i += 1
            rank -= 1
            continue
        end
        piv = pivot_indices[pidx] # the pivot
        (piv != i) && Base.swapcols!(A, piv, i)
        
        p = A[i, j]

        for j′ = (j+1):M
            a_ij = A[i, j′]
            for k = i:N
                A[k, j′] = det2ₘ(-a_ij, -p, A[k, j′], A[k, j])
            end
        end

        i += 1
        j += 1
    end

    return rank
end

@inline function det2(A, m)
    @modops(m)

    return @inbounds det2ₘ(A[1, 1], A[1, 2], A[2, 1], A[2, 2])
end

@inline function det3(A, m)
    @modops(m)

    @inbounds begin
        a11, a21, a31 = A[1, 1], A[2, 1], A[3, 1]
        a12, a22, a32 = A[1, 2], A[2, 2], A[3, 2]
        a13, a23, a33 = A[1, 3], A[2, 3], A[3, 3]
    end

    d1 = det2ₘ(a22, a32, a23, a33)
    d2 = det2ₘ(a12, a32, a13, a33)
    d3 = det2ₘ(a12, a22, a13, a23)

    D = det2ₘ(a11, a21, d2, d1)
    D = det2ₘ(a31, -1, D, d3)
    return D
end

@inline function det4(A, m)
    @modops(m)

    @inbounds begin
        a11, a21, a31, a41 = A[1, 1], A[2, 1], A[3, 1], A[4, 1]
        a12, a22, a32, a42 = A[1, 2], A[2, 2], A[3, 2], A[4, 2]
        a13, a23, a33, a43 = A[1, 3], A[2, 3], A[3, 3], A[4, 3]
        a14, a24, a34, a44 = A[1, 4], A[2, 4], A[3, 4], A[4, 4]
    end

    m24 = det2ₘ(a32, a34, a42, a44)
    m34 = det2ₘ(a33, a34, a43, a44)
    m23 = det2ₘ(a32, a33, a42, a43)
    m14 = det2ₘ(a31, a34, a41, a44)
    m13 = det2ₘ(a31, a33, a41, a43)
    m12 = det2ₘ(a31, a32, a41, a42)

    d1 = det2ₘ(a22, a23, m24, m34)
    d1 = det2ₘ(a24, -1, d1, m23)
    d2 = det2ₘ(a21, a23, m14, m34)
    d2 = det2ₘ(a24, -1, d2, m13)
    d3 = det2ₘ(a21, a22, m14, m24)
    d3 = det2ₘ(a24, -1, d3, m12)
    d4 = det2ₘ(a21, a22, m13, m23)
    d4 = det2ₘ(a23, -1, d4, m12)

    D = det2ₘ(a11, a12, d2, d1) +ₘ det2ₘ(a13, a14, d4, d3)
    return D
end

function maybemod!(A, AI, AI_max, m)
    if AI_max < m
        A .= AI # no mod needed
    else
        T = promote_type(Int, eltype(AI))
        A .= rem.(T.(AI), m)
    end
end

function det_residuals(mods, AI, AI_max)
    A = similar(AI, Float64)

    map(mods) do m
        maybemod!(A, AI, AI_max, m)
        gauss_determinant!(A, m)
    end
end

function perm_residuals(mods, AI, AI_max)
    A = similar(AI, Float64)

    map(mods) do m
        maybemod!(A, AI, AI_max,  m)
        permanentmod(A, m)
    end
end

function maxrank(mods, AI, AI_max)
    A = similar(AI, Float64)

    maximum(mods) do m
        # do a normal integer mod first
        maybemod!(A, AI, AI_max, m)
        rank!(A, m)
    end
end

function compsign(mods, rs, W)
    k = length(mods)

    S_j = zero(eltype(rs))
    for j = k:-1:1
        S_j = sum(i -> fmod(rs[i] * W[i, j], mods[i]) / mods[i], 1:j)
        S_j = S_j - round(S_j) # fractional part
        
        if abs(S_j) > j * eps()
            return S_j
        end
    end
    
    return S_j
end

function try_bareiss(A, bounds)
    if bounds[:lg_bareiss] < log2(typemax(Int))
        A_Int = eltype(A) == Int ? A : Int.(A)
        return LinearAlgebra.det_bareiss!(A_Int)
    else
        return nothing
    end
end

# chinese remainder theorem
# in-place BigInt arithmetic is a lot faster
function crt(residuals, mods)
    n = length(residuals)
    M, V = lagrange(mods)
    modprod = M[n]

    bits = ndigits(modprod; base=2) + trailing_zeros(nextpow(2, n))
    s, tmp = BigInt(; nbits=bits), BigInt(; nbits=bits)
    for i = 1:n
        Base.GMP.MPZ.mul_si!(tmp, V[i, n], residuals[i])
        Base.GMP.MPZ.add!(s, tmp)
    end

    d = Base.GMP.MPZ.tdiv_r!(s, modprod)
    if d < 0
        Base.GMP.MPZ.add!(d, modprod)
    end

    d = (d > modprod ÷ 2) ? (d - modprod) : d
    return d
end

function detsign(AI::AbstractMatrix{T}) where {T <: Integer}
    LinearAlgebra.checksquare(AI)
    size(AI) == 1 && return sign(AI[1])

    bds = bounds(AI)

    # use Bareiss if it won't overflow
    bdet = try_bareiss(AI, bds)
    !isnothing(bdet) && return sign(bdet)

    # add 1 so that the bound is doubled,
    # because determinant can be pos or neg.
    n, mods, W = modsfor(bds[:lg_hadamard] + 1)
    (n == 0) && return 0 # determinant must be zero

    rs = det_residuals((@view mods[1:n]), AI, bds[:max])
    
    S = compsign((@view mods[1:n]), rs, W)
    return Int(sign(S))
end

function detbig(AI::AbstractMatrix{T}) where {T <: Integer}
    LinearAlgebra.checksquare(AI)
    size(AI) == 1 && return big(AI[1])

    bds = bounds(AI)

    bdet = try_bareiss(AI, bds)
    !isnothing(bdet) && return big(bdet)

    n, mods, _ = modsfor(bds[:lg_hadamard] + 1)
    (n == 0) && return big(0)

    rs = det_residuals((@view mods[1:n]), AI, bds[:max])

    return crt(rs, mods)
end

function rank(AI::AbstractMatrix{T}) where {T <: Integer}
    bds = bounds(AI)
    n, mods, _ = modsfor(bds[:lg_hadamard] + 1)

    return maxrank((@view mods[1:n]), AI, bds[:max])
end

function permbig(AI::AbstractMatrix{T}) where {T <: Integer}
    LinearAlgebra.checksquare(AI)

    bds = bounds(AI)
    n, mods, _ = modsfor(bds[:lg_perm] + 1)
    (n == 0) && return big(0)

    rs = perm_residuals((@view mods[1:n]), AI, bds[:max])

    return crt(rs, mods)
end

end # module IntegerMatrixOps
