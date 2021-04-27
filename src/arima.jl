function forecast(model::M, z::AbstractVector{T}) where {p,d,q,T,M <: ARIMA{p,d,q,T}}
    N = length(z)
    arma = ARMA(model.μ, model.ϕ, model.θ, model.σ2)
    ∇z = difference(z, d=d)
    ∇zhat = forecast(arma, ∇z)
    push!(∇z, ∇zhat)
    z = N < d ? fill(model.μ, d) : z
    return last(integrate(∇z, z, d=d))
end