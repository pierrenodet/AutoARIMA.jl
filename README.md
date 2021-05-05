# AutoARIMA.jl

[![Build Status](https://github.com/pierrenodet/AutoARIMA.jl/workflows/CI/badge.svg)](https://github.com/pierrenodet/AutoARIMA.jl/actions?query=workflow%3ACI)
[![codecov.io](https://codecov.io/github/pierrenodet/AutoARIMA.jl/branch/main/graph/badge.svg)](http://codecov.io/github/pierrenodet/AutoARIMA.jl/branch/main)
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://pierrenodet.github.io/AutoARIMA.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://pierrenodet.github.io/AutoARIMA.jl/dev)

Automatic multi-seasonal ARIMA Learning with Box and Jenkins method.

```julia
using AutoARIMA

julia> z = seriesG
144-element StaticArrays.SVector{144, Int64} with indices SOneTo(144):
 112
 118
 132
 129
   ⋮
 508
 461
 390
 432

julia> sarima = fit(MSARIMAParams([0,0],[1,1],[1,1],[1,12]),log.(z))
ARMAModel{13, 13, Float64}(-0.002232109177566169, [1.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, 1.0, -1.0], [0.34919600156185515, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, 0.31610711882734377, -0.11038334195974667], Float64[], 0.0016285014276129083)

julia> forecast(sarima,log.(z))
6.104360295754863

julia> best = boxjenkins(z,true,2,2,2,[1,12],criterium=aicc)
AutoARIMA.ARMAXModel{25, 0, 0, Float64}(-0.0009527614685900121, [1.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0  …  -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -1.0, 1.0], Float64[], Float64[], 0.004332904404300882)
```