using Statistics

function mse(z::AbstractVector, zhat::AbstractVector)
    return mean((z .- zhat).^2)
end

function rmse(z::AbstractVector, zhat::AbstractVector)
    return sqrt(mse(z, zhat))
end

function mae(z::AbstractVector, zhat::AbstractVector)
    return mean(abs.(z .- zhat))
end

function mape(z::AbstractVector, zhat::AbstractVector)
    reduce(&, z .> 0) || throw(DomainError("mape requires strictly positive data"))
    return mean(abs.(z .- zhat) ./ z)
end

function aic(m::M) where {M <: SARIMAX}
    return -2 * log(m.σ2) + 2 * k(m)
end

function aicc(m::M, n::Integer) where {M <: SARIMAX}
    return aic(m) + 2 * k(m) * (k(m) + 1) / (n - k(m) - 1)
end

function bic(m::M, n::Integer) where {M <: SARIMAX}
    return -2 * log(m.σ2) + log(n) * k(m)
end
