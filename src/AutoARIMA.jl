module AutoARIMA

using LinearAlgebra
using Optim
using Polynomials
using RecipesBase
using StaticArrays
using Statistics
using HypothesisTests

export seriesA,seriesB,seriesB2,seriesC,seriesD,seriesE,seriesF,seriesG,dowj,wine,lake
export autocovariance,autocovariance_matrix,autocorrelation,autocorrelation_matrix,partial_autocorrelation
export isinversible, isstationary
export innovations, levinson_durbin, least_squares, yule_walker, hannan_rissanen
export boxcox, guerrero, difference, integrate
export simulate, forecast, fit, toarma, k, residuals
export aic, aicc, bic, mse, rmse, mae, mape
export correlogram,partial_correlogram
export AbstractModel, AbstractParams
export ARParams, MAParams, ARMAParams, ARIMAParams, MSARIMAParams
export ARModel, MAModel, ARMAModel
export MA∞, AR∞

include("datasets.jl")
include("stats.jl")
include("transforms.jl")
include("abstract.jl")
include("criteria.jl")
include("simulate.jl")
include("recipes.jl")
include("ls.jl")
include("arma.jl")
include("ar.jl")
include("ma.jl")
include("arima.jl")
include("sarima.jl")
include("auto.jl")

end