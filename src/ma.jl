using StaticArrays

isstationary(model::MA) = true
    
isinversible(model::MA) = false

function innovations!(θ::AbstractMatrix{T}, σ2::AbstractVector{T}, γ::AbstractVector{T}, q::Integer, m::Integer) where {T}
    θ[m,m] = γ[1 + m] / σ2[1]
    for k in 2:m
        tmp = zero(T)
        for j in 1:k
            tmp += θ[k - 1,(k - 1) - (j - 1) + 1] * θ[m,m - (j - 1)] * σ2[1 + (j - 1)]
        end
        θ[m,m - (k - 1)] =  m - (k - 1) > q ? zero(T) : (one(T) / σ2[1 + (k - 1)]) * (γ[1 + m - (k - 1)] - tmp)
    end
    σ2[m + 1] = γ[1]
    for j in 1:m
        σ2[m + 1] -= θ[m,m - (j - 1)]^2 * σ2[1 + (j - 1)]
    end
end

function innovations(z::AbstractVector, q::Integer; m::Integer=17)
    m >= q || throw(ArgumentError("number of iterations m should be higher than the number of coefficients q"))
    γ = map(i -> autocovariance(z, i), 0:m)
    T = typeof(zero(eltype(z)) / 1)
    σ2 = Vector{T}(undef, m + 1)
    σ2[1] = γ[1]
    θ = zeros(T, m, m)
    for k in 1:m
        innovations!(θ, σ2, γ, q, k)
    end
    return θ[m,1:q], σ2[m + 1]
end