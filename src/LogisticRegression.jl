module LogisticRegression

using Statistics, Optim, NLSolversBase

export logreg

include("structures.jl")
include("estimation.jl")
include("hypothesis_tests.jl")

end # module
