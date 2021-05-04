struct MSARIMAParams{T <: Integer} <: AbstractParams
    c::Bool
    p::Vector{Vector{T}}
    d::Vector{T}
    q::Vector{Vector{T}}
    s::Vector{T}
    function MSARIMAParams(c::Bool, p::Vector{Vector{T}}, d::Vector{T}, q::Vector{Vector{T}}, s::Vector{T}) where T <: Integer
        length(p) == length(q) == length(d) == length(s) || throw(DimensionMismatch("c,p,q,d and s should have the same length"))
        return new{T}(c, p, d, q, s)
    end
end

MSARIMAParams(c::Bool, p::Vector{T},d::Vector{T},q::Vector{T},s::Vector{T}) where T <: Integer = MSARIMAParams(c, map(i -> collect(1:i), p), d, map(i -> collect(1:i), q), s)
MSARIMAParams(p::Vector{T},d::Vector{T},q::Vector{T},s::Vector{T}) where T <: Integer = MSARIMAParams(true, p, d, q, s)

function toarma(params::MSARIMAParams, armas::Vector{ARMAModel}, μ)
    M = length(armas)
    ϕp = Polynomial(1)
    θp = Polynomial(1)
    for i in 1:M
        ϕp *= Polynomial([1;.-armas[i].ϕ]) * Polynomial(1 - variable()^params.s[i])^params.d[i]
        θp *= Polynomial([1;.-armas[i].θ])
    end
    p0 = length(ϕp) - 1
    q0 = length(θp) - 1
    ϕ0 = .-coeffs(ϕp)[2:end]
    μ0 = params.c[1] ? μ * (1 - sum(ϕ0)) : 0.0
    μ0 = isnan(μ0) ? 0.0 : μ0
    return ARMAModel{p0,q0}(μ0, ϕ0, .-coeffs(θp)[2:end], armas[1].σ2)
end

function fit(params::MSARIMAParams, z::AbstractVector; n::Integer=10)
    M = length(params.p)
    N = length(z)
    length(z) > maximum(params.d) * (1 + maximum(params.s)) || throw(DomainError("ARIMA requires at least d+D*s data points"))
    armas = ARMAModel[]
    ∇z = z .- mean(z)
    for i in 1:M
        ∇z = difference(∇z, d=params.d[i], s=params.s[i])
    end
    for i in 1:M
        arma = fit(ARMAParams(false, params.p[i] .* params.s[i], params.q[i] .* params.s[i]), ∇z, n=n)
        push!(armas, arma)
    end
    toarma(params, armas, mean(z))
end