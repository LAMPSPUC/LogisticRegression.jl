module LogisticRegression

using Statistics, Optim

export logreg

include("structures.jl")
include("estimation.jl")
include("hypothesis_tests.jl")

end # module
