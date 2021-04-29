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