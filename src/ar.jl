struct ARParams <: AbstractParams
    c::Bool
    p::Vector{<:Integer}
end

ARParams(c::Bool, p::Integer) = ARParams(c, collect(1:p))
ARParams(p::Integer) = ARParams(true, p)

function fit(params::ARParams, z::AbstractVector{T}) where T
    p = params.p
    P = maximum(p)
    if p == collect(1:P)
        ϕ, σ2 = levinson_durbin(z, P)
        μ = params.c ? mean(z) * (1 - sum(ϕ)) : zero(T)
    else
        μ, ϕ, θ, β, σ2 = least_squares(z, params.c, params.p)
    end
    return ARModel{P}(μ, ϕ, σ2)
end

const ARModel{p,T} = ARMAXModel{p,0,0,T}
ARModel{p}(μ::T,ϕ::AbstractVector{T},σ2::T) where {p,T} = ARMAXModel{p,0,0,T}(μ, ϕ, T[], T[], σ2)

isstationary(model::ARModel) = all(norm.(roots(Polynomial([1;.-model.ϕ]))) .> 1)

isinvertible(model::ARModel) = true

function forecast(model::M, z::AbstractVector{T}) where {p,T,M <: ARModel{p,T}}
    N = length(z)
    zhat = model.μ
    for i in 1:min(N, p)
        zhat += model.ϕ[i] * z[N - i + 1]
    end
    return zhat
end

least_squares(z::AbstractVector{T}, c::Bool, p::Vector{<:Integer}) where T = least_squares(z, T[], zeros(T, 0, 0), c, p, Int[])
least_squares(z::AbstractVector{T}, c::Bool, p::Integer) where T = least_squares(z, c, collect(1:p))
least_squares(z::AbstractVector, p::Integer) = least_squares(z, true, p)

function yule_walker(z::AbstractVector, p::Integer)
    ρ = map(i -> autocorrelation(z, i), 1:p)
    Γ = autocorrelation_matrix(z, p)
    ϕ = cholesky(Γ) \ ρ
    m = mean(z)
    γ0 = autocovariance(z, 0) 
    σ2 = (1 - dot(ϕ, ρ)) * γ0
    return ϕ, σ2
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
    return ϕ, σ2[]
end