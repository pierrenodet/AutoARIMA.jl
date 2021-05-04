struct ARIMAParams{T} <: AbstractParams
    c::Bool
    p::Vector{T}
    d::T
    q::Vector{T}
    function ARIMAParams(c::Bool, p::AbstractVector{T}, d::T, q::AbstractVector{T}) where T <: Integer
        return new{T}(c, p, d, q)
    end
end

ARIMAParams(c::Bool,p::Integer,d::Integer,q::Integer) = ARIMAParams(c, collect(1:p), d, collect(1:q))
ARIMAParams(p::Integer,d::Integer,q::Integer) = ARIMAParams(true, p, d, q)

function toarma(params::ARIMAParams, arma::ARMAModel, μ)
    P = isempty(params.p) ? 0 : maximum(params.p)
    Q = isempty(params.q) ? 0 : maximum(params.q)
    d = params.d
    ϕdp = Polynomial([1;.-arma.ϕ]) * Polynomial(1 - variable())^d
    ϕ0 = .-coeffs(ϕdp)[2:end]
    μ0 = params.c ? μ * (1 - sum(ϕ0)) : zero(typeof(μ))
    return ARMAModel{P + d,Q}(μ0, ϕ0, arma.θ, arma.σ2)
end

toarma(params::ARIMAParams, arma::ARMAModel) = toarma(params, arma, arma.μ)

function fit(params::ARIMAParams, z::AbstractVector; n::Integer=2)
    length(z) > params.d || throw(DomainError("ARIMA requires at least d data points"))
    arima = fit(ARMAParams(params.c, params.p, params.q),  difference(z, d=params.d), n=n)
    return toarma(params, arima)
end