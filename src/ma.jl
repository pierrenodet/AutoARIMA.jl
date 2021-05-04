struct MAParams <: AbstractParams
    q::Vector{<:Integer}
end

MAParams(q::Integer) = MAParams(collect(1:q))

function fit(params::MAParams, z::AbstractVector{T}; n::Integer=20) where T
    q = params.q
    Q = maximum(q)
    if q == collect(1:Q)
        μ = mean(z)
        z = z .- μ
        θ, σ2 = innovations(z, Q, n=n)
        ma = MAModel{Q}(μ, θ, σ2)
    else
        ma = fit(ARMAParams(false, eltype(q)[], q), z, n=n)
    end
    return ma
end

const MAModel{q,T} = ARMAXModel{0,q,0,T}
MAModel{q}(μ::T,θ::AbstractVector{T},σ2::T) where {q,T} = ARMAXModel{0,q,0,T}(μ, T[], θ, T[], σ2)

isstationary(model::MAModel) = true

isinvertible(model::MAModel) = all(norm.(roots(Polynomial([1;.-model.θ]))) .> 1)

function innovations!(θ::AbstractMatrix{T}, σ2::AbstractVector{T}, γ::AbstractVector{T}, q::Integer, n::Integer) where {T}
    θ[n,n] = γ[1 + n] / σ2[1]
    for k in 2:n
        tmp = zero(T)
        for j in 1:k
            tmp += θ[k - 1,(k - 1) - (j - 1) + 1] * θ[n,n - (j - 1)] * σ2[1 + (j - 1)]
        end
        θ[n,n - (k - 1)] =  n - (k - 1) > q ? zero(T) : (-one(T) / σ2[1 + (k - 1)]) * (γ[1 + n - (k - 1)] - tmp)
    end
    σ2[n + 1] = γ[1]
    for j in 1:n
        σ2[n + 1] -= θ[n,n - (j - 1)]^2 * σ2[1 + (j - 1)]
    end
end

function innovations(z::AbstractVector, q::Integer; n::Integer=20)
    n >= q || throw(ArgumentError("number of iterations should be higher than the number of coefficients q"))
    γ = map(i -> autocovariance(z, i), 0:n)
    T = typeof(zero(eltype(z)) / 1)
    σ2 = Vector{T}(undef, n + 1)
    σ2[1] = γ[1]
    θ = zeros(T, n, n)
    for k in 1:n
        innovations!(θ, σ2, γ, q, k)
    end
    return θ[n,1:q], σ2[n + 1]
end