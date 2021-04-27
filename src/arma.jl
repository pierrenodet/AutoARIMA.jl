using StaticArrays, LinearAlgebra

function hannan_rissanen(z::AbstractVector, p::Integer, q::Integer; m::Integer=20)
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
    θ = reverse(ϕ[p + 2:end])
    ϕ = reverse(ϕ[2:p + 1])
    return μ, ϕ, θ, σ2
end

function forecast(model::M, z::AbstractVector{T}) where {p,q,T,M <: ARMA{p,q,T}}
    N = length(z)
    a = Vector{T}(undef, N + 1)
    a[1] = zero(T)
    for i in 1:N
        a[i + 1] = z[i] - model.μ
        for j in 1:min(q, i)
            a[i + 1] -= model.θ[j] * a[i - j + 1]
        end
        for j in 1:min(p, i - 1)
            a[i + 1] -= model.ϕ[j] * z[i - j]
        end
    end
    zhat = model.μ
    for j in 1:min(N, q)
        zhat += model.θ[j] * a[N - j + 2]
    end
    for j in 1:min(N, p)
        zhat += model.ϕ[j] * z[N - j + 1]
    end
    return zhat
end

function MA(model::M, m::Integer) where {p,q,T,M <: ARMA{p,q,T}}
    ψ = Vector{T}(undef, m)
    for j in 1:m
        ψ[j] = (j > q ? zero(T) : model.θ[j])
        for k in 1:p
            ψ[j] += model.ϕ[k] * (j - k == 0 ? one(T) : j - k < 0 ? zero(T) : ψ[j - k])
        end
    end
    return MA(model.μ / (1 - sum(model.ϕ)), SVector{m,T}(ψ), model.σ2)
end

function AR(model::M, m::Integer) where {p,q,T,M <: ARMA{p,q,T}}
        π = Vector{T}(undef, m)
    for j in 1:m
        π[j] = -(j > p ? zero(T) : model.ϕ[j])
        for k in 1:q
            π[j] += - model.θ[k] * (j - k == 0 ? one(T) : j - k < 0 ? zero(T) : π[j - k])
        end
    end
    return AR(model.μ / (1 - sum(model.ϕ)) * (1 + sum(π)), SVector{m,T}(π), model.σ2)
end