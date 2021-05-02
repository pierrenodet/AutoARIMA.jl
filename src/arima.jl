struct ARIMAParams{T} <: AbstractParams
    p::Vector{T}
    d::T
    q::Vector{T}
    function ARIMAParams(p::AbstractVector{T}, d::T, q::AbstractVector{T}) where T <: Integer
        return new{T}(p, d, q)
    end
end

ARIMAParams(c::Bool,p::Integer,d::Integer,q::Integer) = ARIMAParams(collect(!c:p), d, collect(1:q))
ARIMAParams(p::Integer,d::Integer,q::Integer) = ARIMAParams(true, p, d, q)

function toarma(params::ARIMAParams, arma::ARMAModel{p,q,T}) where {p,q,T}
    d = params.d
    ϕdp = Polynomial([1;.-arma.ϕ]) * Polynomial(1 - variable())^d
    ϕ0 = .-coeffs(ϕdp)[2:end]
    μ0 = arma.μ
    return ARMAModel{p + d,q}(μ0, ϕ0, arma.θ, arma.σ2)
end

function fit(params::ARIMAParams, z::AbstractVector)
    length(z) > params.d || throw(DomainError("ARIMA requires at least d data points"))
    arima = fit(ARMAParams(params.p, params.q),  difference(z, d=params.d))
    return toarma(params, arima)
end