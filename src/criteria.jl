function mse(m::M, z::AbstractVector{T}) where {T,M <: AbstractModel{T}}
    return mean(residuals(m, z).^2)
end

function rmse(m::M, z::AbstractVector{T}) where {T,M <: AbstractModel{T}}
    return sqrt(residuals(m, z))
end

function mae(m::M, z::AbstractVector{T}) where {T,M <: AbstractModel{T}}
    return mean(residuals(m, z))
end

function mape(m::M, z::AbstractVector{T}) where {T,M <: AbstractModel{T}}
    reduce(&, z .> 0) || throw(DomainError("mape requires strictly positive data"))
    return mean(abs.(residuals(m, z)) ./ z)
end

function aic(m::M) where {M <: AbstractModel}
    return -2 * log(m.σ2) + 2 * k(m)
end

function aicc(m::M, z::AbstractVector) where {M <: AbstractModel}
    return aic(m) + 2 * k(m) * (k(m) + 1) / (length(z) - k(m) - 1)
end

function bic(m::M, z::AbstractVector) where {M <: AbstractModel}
    return -2 * log(m.σ2) + log(length(z)) * k(m)
end
