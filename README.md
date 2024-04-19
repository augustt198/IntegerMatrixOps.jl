# IntegerMatrixOps.jl

Fast and exact operations for integer matrices.

This solves the problem of `LinearAlgebra.det(::Matrix{Int})`
being fast but potentially incorrect, while
`LinearAlgebra.det(::Matrix{BigInt})` is correct but slow.

```julia
using IntegerMatrixOps, LinearAlgebra, BenchmarkTools

A = rand(Int, 15, 15)

@btime IntegerMatrixOps.detbig($A);
#  28.750 μs (130 allocations: 9.62 KiB)

@btime LinearAlgebra.det(Abig) setup=(Abig=big.(A));
#  320.375 μs (11171 allocations: 393.80 KiB)

detbig(A) == det(big.(A))
# true

# if only the sign is wanted
@btime IntegerMatrixOps.detsign($A);
#  25.500 μs (2 allocations: 2.34 KiB)
```

This example shows how computing the
determinant with floating point can
cause sign errors:

```julia
# nearly singular
A = fill(typemax(Int), 10, 10) - I

det(A) # => 0.0 (incorrect)
detsign(A) # => -1 (correct)

rank(A) # => 1 (incorrect)
IntegerMatrixOps.rank(A) # => 10 (correct)
```

## Functions

- `detsign(A)` exact sign of derminant of `A` (`-1`, `0`, or `+1`)
- `detbig(A)` exact determinant returned as a `BigInt`
- `permbig(A)` exact perminant returned as a `BigInt`
