using Documenter, LogisticRegression

makedocs(;
    modules=[LogisticRegression],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
    ],
    repo="https://github.com/LAMPSPUC/LogisticRegression.jl/blob/{commit}{path}#L{line}",
    sitename="LogisticRegression.jl",
    authors="Gabrile Mizuno <gabrielmizuno@gmail.com>",
    assets=String[],
)

deploydocs(;
    repo="github.com/LAMPSPUC/LogisticRegression.jl",
)
