module AutoARIMA

export seriesA,seriesB,seriesB2,seriesC,seriesD,seriesE,seriesF,seriesG
export autocovariance,autocovariance_matrix,autocorrelation,autocorrelation_matrix,partial_autocorrelation,correlogram,partial_correlogram
export isinversible,isstationary,inverse
export innovations, levinson_durbin, least_squares, yule_walker, hannan_rissanen, auto_sarimax
export boxcox, guerrero, difference, integrate
export simulate, forecast
export aic, aicc, bic, mse, rmse, mae, mape
export AR,MA,ARMA,ARIMA,SARIMA

include("stats.jl")
include("sarima.jl")
include("ar.jl")
include("ma.jl")
include("arma.jl")
include("arima.jl")
include("transforms.jl")
include("criteria.jl")
include("datasets.jl")

end