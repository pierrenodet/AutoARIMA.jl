using LinearAlgebra, Statistics

isstationary(model::AR) = false

isinversible(model::AR) = true
    
function least_squares(z::AbstractVector{T}, p::Integer) where {T}
    N = length(z)
    Z = Matrix{T}(undef, N - p, p + 1)
    Z[:,1] .= one(T)
    for j in 2:p + 1
        Z[:,j] = view(z, j - 1:N - p + j - 2)
    end
    ϕ = Z \ view(z, p + 1:N)
    μ = ϕ[1]
    ε = view(z, p + 1:N) - Z * ϕ
    σ2 = dot(ε, ε) / (N - p)
    return μ, reverse(view(ϕ, 2:p + 1)), σ2
end

function yule_walker(z::AbstractVector, p::Integer)
    ρ = map(i -> autocorrelation(z, i), 1:p)
    Γ = autocorrelation_matrix(z, p)
    ϕ = cholesky(Γ) \ ρ
    m = mean(z)
    μ = m * (1 - sum(ϕ))
    γ0 = autocovariance(z, 0) 
    σ2 = (1 - dot(ϕ, ρ)) * γ0
    return μ, ϕ, σ2
end

function levinson_durbin!(ϕ::AbstractVector{T}, σ2::Ref{T}, ρ::AbstractVector{T}, p::Integer) where {T}
    num = ρ[p]
    den = one(T)
    for j in 1:p - 1
        num -= ϕ[j] * ρ[p - j]
        den -= ϕ[j] * ρ[j]
    end
    ϕ[p] = num / den
    for j in 1:div(p - 1, 2)
        tmp = ϕ[j]
        ϕ[j] -= ϕ[p] * ϕ[p - j]
        ϕ[p - j] -= ϕ[p] * tmp
    end
    if isodd(p - 1) ϕ[div(p - 1, 2) + 1] *= (one(T) - ϕ[p]) end
    σ2[] *= (one(T) - ϕ[p]^2)
end

function levinson_durbin(z::AbstractVector, p::Integer)
    ρ = map(i -> autocorrelation(z, i), 1:p)
    T = typeof(zero(eltype(z)) / 1)
    ϕ = zeros(T, p)
    σ2 = Ref{T}(autocovariance(z, 0))
    for i in 1:p
        levinson_durbin!(ϕ, σ2, ρ, i)
    end
    μ = mean(z) * (1 - sum(ϕ))
    return μ, ϕ, σ2[]
end

# function burg!(ϕ::AbstractVector{T}, σ2::Ref{T}, f::AbstractVector{T}, b::AbstractVector{T}, p::Integer) where {T}
#     N = length(f)
#     num = zero(T)
#     den = zero(T)
#     for i in 1:N - p
#         num += -2 * f[i + p] * b[i]
#         den += f[i + p]^2 + b[i]^2
#     end
#     μ = num / den
#     ϕ[p] = μ
#     for j in 1:div(p - 1, 2)
#         tmp = ϕ[j]
#         ϕ[j] += μ * ϕ[p - j]
#         ϕ[p - j] += μ * tmp
#     end
#     if isodd(p - 1) ϕ[div(p - 1, 2) + 1] *= (one(T) + μ) end
#     for i in 1:N - p
#         tmp = f[i + p]
#         f[i + p] = f[i + p] + μ * b[i]
#         b[i] = b[i] + μ * tmp
#     end
#     σ2[] *= (one(T) - μ^2)
# end

# function burg!(ϕ::AbstractVector{T}, σ2::Ref{T}, f::AbstractVector{T}, b::AbstractVector{T}, D::Ref{T}, N::Integer, p::Integer) where {T}
#     μ = zero(T)
#     for i in 1:N - p
#         μ += f[i + p] * b[i]
#     end
#     μ *= -2 / D[]
#     for j in 1:div(p, 2)
#         tmp = ϕ[j]
#         ϕ[j] += μ * ϕ[p - j]
#         ϕ[p - j] += μ * tmp
#     end
#     if isodd(p) ϕ[div(p, 2) + 1] *= (one(T) + μ) end
#     for i in 1:N - p
#         tmp = f[i + p]
#         f[i + p] = f[i + p] + μ * b[i]
#         b[i] = b[i] + μ * tmp
#     end
#     D[] = (one(T) - μ^2) * D[] - f[p]^2 - b[N - p]^2
#     σ2[] *= (one(T) - μ^2)
# end


# function burg(z::AbstractVector, p::Integer)
#     N = length(z)
#     T = typeof(zero(eltype(z)) / 1)
#     f = T[z...]
#     b = T[z...]
#     ϕ = zeros(T, p)
#     σ2 = Ref{T}(autocovariance(z, 0))
#     for i in 1:p
#         burg!(ϕ, σ2, f, b, i)
#     end
#     μ = mean(z) * (1 - sum(ϕ))
#     return μ, ϕ, σ2[]
# end

function forecast(model::M, z::AbstractVector{T}) where {p,T,M <: AR{p,T}}
    zhat = model.μ
    N = length(z)
    for i in 1:min(N, p)
        zhat += model.ϕ[i] * z[N - i + 1]
    end
    return zhat
end