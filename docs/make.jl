using Documenter, AutoARIMA

makedocs(
    sitename="AutoARIMA.jl"
)

deploydocs(
    devbranch = "main",
    repo = "github.com/pierrenodet/AutoARIMA.jl.git",
)