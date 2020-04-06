using Documenter, LogisticRegression
using Plots

makedocs(;
    modules=[LogisticRegression],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
    ],
    repo="https://github.com/LAMPSPUC/LogisticRegression.jl/blob/{commit}{path}#L{line}",
    sitename="LogisticRegression.jl",
    authors="André Gutierez, Ana Carolina Freire, Gabrile Mizuno, Guilhemer Bodin",
    assets=String[],
)

deploydocs(;
    repo="github.com/LAMPSPUC/LogisticRegression.jl",
)
