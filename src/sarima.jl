struct MSARIMAParams{T <: Integer} <: AbstractParams
    p::Vector{Vector{T}}
    d::Vector{T}
    q::Vector{Vector{T}}
    s::Vector{T}
    function MSARIMAParams(p::Vector{Vector{T}}, d::Vector{T}, q::Vector{Vector{T}}, s::Vector{T}) where T <: Integer
        length(p) == length(q) == length(d) == length(s) || throw(DimensionMismatch("p,q,d and s should have the same length"))
        return new{T}(p, d, q, s)
    end
end

MSARIMAParams(c::Bool, p::Vector{T},d::Vector{T},q::Vector{T},s::Vector{T}) where T <: Integer = MSARIMAParams(map(i -> collect(!c:i), p), d, map(i -> collect(1:i), q), s)
MSARIMAParams(p::Vector{T},d::Vector{T},q::Vector{T},s::Vector{T}) where T <: Integer = MSARIMAParams(false, p, d, q, s)

function toarma(params::MSARIMAParams, armas::Vector{ARMAModel})
    M = length(armas)
    ϕ0 = Polynomial(1)
    θ0 = Polynomial(1)
    for i in 1:M
        ϕ0 *= Polynomial([1;.-armas[i].ϕ]) * Polynomial(1 - variable()^params.s[i])^params.d[i]
        θ0 *= Polynomial([1;.-armas[i].θ])
    end
    p0 = length(ϕ0) - 1
    q0 = length(θ0) - 1
    return ARMAModel{p0,q0}(armas[1].μ, .-coeffs(ϕ0)[2:end], .-coeffs(θ0)[2:end], armas[1].σ2)
end

function fit(params::MSARIMAParams, z::AbstractVector; n::Integer=10)
    M = length(params.p)
    length(z) > maximum(params.d) * (1 + maximum(params.s)) || throw(DomainError("ARIMA requires at least d+D*s data points"))
    armas = ARMAModel[]
    for i in 1:M
        arma = fit(ARMAParams(params.p[i] .* params.s[i], params.q[i] .* params.s[i]), difference(z, d=params.d[i], s=params.s[i]), n=n)
        push!(armas, arma)
    end
    arma = fit(ARMAParams(params.p[1] .* params.s[1], params.q[1] .* params.s[1]), difference(z, d=params.d[1], s=params.s[1]), n=n)
    return toarma(params, armas)
end