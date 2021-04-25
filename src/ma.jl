using StaticArrays

isstationary(model::MA) = true
    
isinversible(model::MA) = false

function innovations!(θ::AbstractMatrix{T}, σ2::AbstractVector{T}, γ::AbstractVector{T}, m::Integer) where {T}
    θ[m,m] = γ[1 + m] / σ2[1]
    for k in 2:m
        tmp = zero(T)
        for j in 1:k
            tmp += θ[k - 1,(k - 1) - (j - 1) + 1] * θ[m,m - (j - 1)] * σ2[1 + (j - 1)]
        end
        θ[m,m - (k - 1)] = (one(T) / σ2[1 + (k - 1)]) * (γ[1 + m - (k - 1)] - tmp)
    end
    σ2[m + 1] = γ[1]
    for j in 1:m
        σ2[m + 1] -= θ[m,m - (j - 1)]^2 * σ2[1 + (j - 1)]
    end
end

function innovations(z::AbstractVector, q::Integer; m::Integer=0)
    m = max(m, q)
    γ = map(i -> autocovariance(z, i), 0:m)
    T = typeof(zero(eltype(z)) / 1)
    σ2 = Vector{T}(undef, m + 1)
    σ2[1] = γ[1]
    θ = zeros(T, m, m)
    for k in 1:m
        innovations!(θ, σ2, γ, k)
    end
    return θ[m,1:q], σ2[m + 1]
end

function forecast(model::M, z::AbstractVector{T}) where {q,T,M <: MA{q,T}}
    N = length(z)
    a = Vector{T}(undef, N + 1)
    a[1] = model.μ
    for i in 1:N
        a[i + 1] = z[i]
        for j in 1:min(q, i)
            a[i + 1] -= model.θ[j] * a[i - j + 1]
        end
    end
    zhat = zero(T)
    for j in 1:min(N, q)
        zhat += model.θ[j] * a[N - j + 2]
    end
    return zhat
end

forecast(model::M) where {q,T,M <: MA{q,T}} = forecast(model, T[])