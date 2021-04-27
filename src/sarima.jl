function auto_sarimax(z::AbstractVector{T}, p::Integer, d::Integer, q::Integer, P::Integer, D::Integer, Q::Integer, s::Integer) where T
    N = length(z)
    N > d + D*s || throw(DomainError("SARIMAX d D s requires at least d + D*s data points"))
    ∇sdz = difference(z, d=D, s=s)
    Μ, Φ, Θ, Σ2 = hannan_rissanen(∇sdz, P, Q)
    μ, ϕ, θ, σ2 = hannan_rissanen(difference(∇sdz, d=d), p, q)
    return SARIMAX{d,D,s}(μ, SVector{p,T}(ϕ), SVector{q,T}(θ), SVector{P,T}(Φ), SVector{Q,T}(Θ), σ2)
end

function forecast(model::M, z::AbstractVector{T}) where {p,d,q,P,D,Q,s,T,M <: SARIMAX{p,d,q,P,D,Q,s,T}}
    N = length(z)
    arma = ARMA(model.μ, model.ϕ, model.θ, model.σ2)
    sarma = ARMA(model.μ, model.Φ, model.Θ, model.σ2)
    ∇s = difference(z, d=D, s=s)
    ∇shat = forecast(sarma, ∇s)
    push!(∇s, ∇shat)
    i∇s = integrate(∇s, z, d=D, s=s)
    ∇z = difference(i∇s, d=d)
    ∇zhat = forecast(arma, ∇z)
    push!(∇z, ∇zhat)
    return last(integrate(∇z, z, d=d))
end