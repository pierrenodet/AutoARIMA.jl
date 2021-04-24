using Statistics, LinearAlgebra

function autocovariance(z::AbstractVector{T}, k::Integer) where {T}
    N = length(z)
    m = mean(z)
    γ = zero(T)
    @inbounds @simd for i in 1:N - k
        γ += (z[i + k] - m) * (z[i] - m)
    end
    return γ / N
end

function autocorrelation(z::AbstractVector, k::Integer)
    return autocovariance(z, k) / autocovariance(z, 0)
end

function autocovariance_matrix(z::AbstractVector, k::Integer)
    N = length(z)
    T = typeof(zero(eltype(z)) / 1)
    γ = Vector{T}(undef, k)
    for i in 1:k
        γ[i] = autocovariance(z, i - 1)
    end
    Γ = Matrix{T}(undef, k, k)
    for j in 1:k
        for i in 1:k
            Γ[i,j] = γ[abs(i - j) + 1]
        end
    end
    return Γ
end

function autocorrelation_matrix(z::AbstractVector, k::Integer)
    return autocovariance_matrix(z, k) ./ autocovariance(z, 0)
end

function partial_autocorrelation(z::AbstractVector, k::Integer)
    Ρ = autocorrelation_matrix(z, k)
    Ρstar = copy(Ρ)
    Ρstar[1:k - 1,k] = Ρstar[2:k,1]
    Ρstar[k,k] = autocorrelation(z, k)
    return det(Ρstar) / det(cholesky(Symmetric(Ρ)))
end

function correlogram(z::AbstractVector, k::Integer)
    N = length(z)
    return -1 / √N, map(i -> autocorrelation(z, i), 0:k), 1 / √N
end

function partial_correlogram(z::AbstractVector, k::Integer; recursive::Bool=true)
    N = length(z)
    if recursive
        ρ = map(i -> autocorrelation(z, i), 1:k)
        T = typeof(zero(eltype(z)) / 1)
        ϕ = zeros(T, k)
        ϕpp = zeros(T, k)
        σ2 = Ref{T}(autocovariance(z, 0))
        for i in 1:k
            levinson_durbin!(ϕ, σ2, ρ, i)
            ϕpp[i] = ϕ[i]
        end
    else
        ϕpp = map(i -> partial_autocorrelation(z, i), 1:k)
    end
    return -1 / √N, ϕpp, 1 / √N
end

function boxcox(z, λ)
	if λ == 0
		return log.(z) 
	else
		return (z.^λ .- 1) ./ λ
	end
end