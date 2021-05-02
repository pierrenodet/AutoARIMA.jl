abstract type AbstractModel{T} end

abstract type AbstractParams end

function k(m::M) where {M <: AbstractModel}
    return sum(m.ϕ .!= 0) + sum(m.θ .!= 0)
end

forecast(model::M) where {T,M <: AbstractModel{T}} = forecast(model, T[])

function residuals!(α::AbstractVector{T}, model::M, z::AbstractVector{T}) where {T,M <: AbstractModel{T}}
    length(α) == length(z) || throw(DimensionMismatch("α should have the same length as z"))
    N = length(z)
    α[1] = z[1] - forecast(model)
    for i in 2:N
        α[i] = z[i] - forecast(model, view(z, 1:i - 1))
    end
    return α
end

residuals(model::M, z::AbstractVector{T}) where {T,M <: AbstractModel{T}} = residuals!(Vector{T}(undef, length(z)), model, z)