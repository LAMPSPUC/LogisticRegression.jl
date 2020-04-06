module LogisticRegression

using Statistics, Optim, NLSolversBase, Distributions

export logreg

include("structures.jl")
include("estimation.jl")
include("hypothesis_tests.jl")

end # module
