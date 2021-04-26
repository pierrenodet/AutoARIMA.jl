using StaticArrays, LinearAlgebra

function hannan_rissanen(z::AbstractVector, p::Integer, q::Integer, m::Integer=20)
    m >= 0 || throw(ArgumentError("order of first ar model should be positive"))
    T = typeof(zero(eltype(z)) / 1)
    N = length(z)
    m = m + max(p, q)
    μinf, ϕinf, σ2inf = levinson_durbin(z, m)
    ar = AR(μinf, SVector{m}(ϕinf), σ2inf)
    a = Vector{T}(undef, N - m)
	for i in 1:(N - m)
		a[i] = z[i + m] - forecast(ar, view(z, 1:i + m - 1))
	end
    Z = Matrix{T}(undef, N - m - q, p + q + 1)
    Z[:,1] .= one(T)
    for j in 1:p
        Z[:,j + 1] = view(z, j + m - p + q:N + j - p - 1)
    end
    for j in 1:q
        Z[:,j + p + 1] = view(a, j:N - m + j - q - 1)
    end
    ϕ = Z \ view(z, m + q + 1:N)
    ε = view(z, m + q + 1:N) - Z * ϕ
    σ2 = dot(ε, ε) / (N - m - q)
    μ = ϕ[1]
    θ = reverse(view(ϕ, p + 2:q + p + 1))
    ϕ = reverse(view(ϕ, 2:p + 1))
    return μ, ϕ, θ, σ2
end

function forecast(model::M, z::AbstractVector{T}) where {p,q,T,M <: ARMA{p,q,T}}
    ar = AR(0.0, model.ϕ, 0.0)
    ma = MA(0.0, model.θ, 0.0)
    return forecast(ma, z) + forecast(ar, z) + model.μ
end

forecast(model::M) where {p,q,T,M <: ARMA{p,q,T}} = forecast(model, T[])

function MA(model::M, m::Integer) where {p,q,T,M <: ARMA{p,q,T}}
    ψ = Vector{T}(undef, m)
    for j in 1:m
        for i in 1:p
            ψ[j] = (j > q ? zero(T) : model.θ[j]) + model.ϕ[i] * (j - i == 0 ? one(T) : j - i < 0 ? zero(T) : ψ[j - i])
        end
    end
    return MA(model.μ, SVector{m,T}(ψ), model.σ2)
end

function AR(model::M, m::Integer) where {p,q,T,M <: ARMA{p,q,T}}
        π = Vector{T}(undef, m)
    for j in 1:m
        for i in 1:q
            π[j] = -(j > p ? zero(T) : model.ϕ[j]) - model.θ[i] * (j - i == 0 ? one(T) : j - i < 0 ? zero(T) : π[j - i])
        end
    end
    return AR(model.μ, SVector{m,T}(π), model.σ2)
end