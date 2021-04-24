module AutoARIMA

export seriesA,seriesB,seriesB2,seriesC,seriesD,seriesE,seriesF
export autocovariance,autocovariance_matrix,autocorrelation,autocorrelation_matrix,partial_autocorrelation,correlogram,partial_correlogram
export isinversible,isstationary,inverse
export innovations, empirical, levinson_durbin, least_squares, yule_walker, hannan_rissanen
export forecast
export AR,MA,ARMA,ARIMA,SARIMAX

include("stats.jl")
include("sarimax.jl")
include("ar.jl")
include("ma.jl")
include("arma.jl")
include("arima.jl")
include("datasets.jl")

end