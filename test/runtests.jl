using IntegerMatrixOps

using Test
using Random
using LinearAlgebra
using Combinatorics

function testdet(A)
    d = LinearAlgebra.det_bareiss(big.(A))
    @test detsign(A) == sign(d)
    @test detbig(A) == d
end

function testperm(A)
    N = size(A, 1)

    # naive but slightly optimized
    tmp = big(0)
    perm = foldl(permutations(1:N); init=big(0)) do sum, σ
        Base.GMP.MPZ.set_si!(tmp, 1)
        p = foldl(enumerate(σ); init=tmp) do prod, idx
            Base.GMP.MPZ.mul_si!(prod, A[idx...])
        end
        Base.GMP.MPZ.add!(sum, p)
    end

    @test permbig(A) == perm
end

Random.seed!(1234)
@testset "random" begin
    for N = (2, 3, 4, 5, 10, 15, 20, 30, 50)
        for T = (Int32, Int64, Bool, UInt8)
            for i = 1:10
                testdet(rand(T, N, N))
            end
        end
    end
end

@testset "singular" begin
    A = rand(Int, 10, 10)
    A[3,:] .= 0
    testdet(A)

    A = rand(Int, 10, 10)
    A[:,3] .= 0
    testdet(A)

    A = rand(Int, 10, 10)
    A[3,:] .= A[4,:] .+ 1 # very nearly singular
    testdet(A)

    A = rand(Int, 10, 10)
    A[:,3] = A[:,4] .+ 1 # very nearly singular
    testdet(A)

    A = rand(Int, 10, 10)
    A[:,3] = A[:,4]
    testdet(A)

    A = rand(Int, 10, 10)
    A[3,:] = A[4,:]
    testdet(A)

    A = rand(Int, 10, 10)
    A[3,:] .= A[4,:] # very nearly singular
    A[3,3] += 1
    testdet(A)
end

@testset "rank" begin
    for N = 1:25
        A = fill(typemax(Int), N, N)
        @test IntegerMatrixOps.rank(A - I) == N

        A[1:N÷2, 1:N÷2] -= I
        @test IntegerMatrixOps.rank(A) == N÷2 + 1
    end
end

@testset "permanent" begin
    for N = (2, 3, 4, 5, 10)
        for T = (Int32, Int64, Bool, UInt8)
            testperm(rand(T, N, N))
        end
    end
end

@testset "inference" begin
    @inferred detsign(rand(Int, 3, 3))
    @inferred detbig(rand(Int, 3, 3))
end
