using Documenter, AutoARIMA

makedocs(
    sitename="AutoARIMA"
)

deploydocs(
    devbranch = "main",
    repo = "github.com/pierrenodet/AutoARIMA.jl.git",
)