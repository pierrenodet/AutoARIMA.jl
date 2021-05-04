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

function toarma(params::ARIMAParams, arma::ARMAModel)
    P = isempty(params.p) ? 0 : maximum(params.p)
    Q = isempty(params.q) ? 0 : maximum(params.q)
    d = params.d
    ϕdp = Polynomial([1;.-arma.ϕ]) * Polynomial(1 - variable())^d
    ϕ0 = .-coeffs(ϕdp)[2:end]
    μ0 = c ? arma.μ /(1-sum(arma.ϕ))*(1-sum(ϕ0)) : zero(T)
    μ0 = isnan(μ0) ? 0 : μ0
    return ARMAModel{P + d,Q}(μ0, ϕ0, arma.θ, arma.σ2)
end

function fit(params::ARIMAParams, z::AbstractVector)
    length(z) > params.d || throw(DomainError("ARIMA requires at least d data points"))
    P = isempty(params.p) ? 0 : maximum(params.p)
    Q = isempty(params.q) ? 0 : maximum(params.q)
    arma = ARMAModel{P,Q}(hannan_rissanen(difference(z, d=params.d), z, params.c, params.p, params.q, n=n)...)
    return toarma(params, arima)
end