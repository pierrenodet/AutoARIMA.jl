struct ARMAParams{T} <: AbstractParams
    c::Bool
    p::Vector{T}
    q::Vector{T}
    function ARMAParams(c::Bool, p::AbstractVector{T}, q::AbstractVector{T}) where T <: Integer
        return new{T}(c, p, q)
    end
end

ARMAParams(c::Bool,p::Integer,q::Integer) = ARMAParams(c, collect(1:p), collect(1:q))
ARMAParams(p::Integer,q::Integer) = ARMAParams(true, p, q)

function fit(params::ARMAParams, z::AbstractVector; n::Integer=2)
    P = isempty(params.p) ? 0 : maximum(params.p)
    Q = isempty(params.q) ? 0 : maximum(params.q)
    return ARMAModel{P,Q}(hannan_rissanen(z, params.c, params.p, params.q, n=n)...)
end
    
struct ARMAXModel{p,q,X,T} <: AbstractModel{T}
    μ::T
    ϕ::SVector{p,T}
    θ::SVector{q,T}
    β::SVector{X,T}
    σ2::T
    function ARMAXModel{p,q,X,T}(μ::T, ϕ::AbstractVector{T}, θ::AbstractVector{T}, β::AbstractVector{T}, σ2::T) where {p,q,X,T}
        return new{p,q,X,T}(μ, SVector{p,T}(ϕ), SVector{q,T}(θ), SVector{X,T}(β), σ2)
    end
end

const ARMAModel{p,q,T} = ARMAXModel{p,q,0,T}
ARMAModel{p,q}(μ::T,ϕ::AbstractVector{T},θ::AbstractVector{T},σ2::T) where {p,q,T} = ARMAXModel{p,q,0,T}(μ, ϕ, θ, T[], σ2)

function hannan_rissanen(z::AbstractVector, c::Bool, p::AbstractVector{<:Integer}, q::AbstractVector{<:Integer}; m::Integer=20, n::Integer=10)
    m >= 0 || throw(ArgumentError("order of first ar model should be positive"))
    T = typeof(zero(eltype(z)) / 1)
    N = length(z)
    P = isempty(p) ? 0 : maximum(p)
    Q = isempty(q) ? 0 : maximum(q)
    m = m + max(P, Q)
    ar = fit(ARParams(true, m), z)
    a = residuals(ar, z)[m + 1:N]
    x = zeros(T, 0, 0)
    Z, ztp1 = ls_matrix(z[m + 1:N], a, x, c, p, q)
    ϕ0 = Z \ ztp1
    ε = ztp1 - Z * ϕ0
    σ2 = dot(ε, ε) / (N - m - Q)
    μ, ϕ, θ, β = ls_params(ϕ0, c, p, q)
    σ2prev = Inf
    for k in 2:n
        arma = ARMAModel{P,Q}(μ, ϕ, θ, σ2)
        a = residuals(arma, z)[m + 1:N]
        ls_matrix!(Z, z[m + 1:N], a, x, c, p, q)
        ϕθ = Z \ ztp1
        ε = ztp1 - Z * ϕθ
        σ2prev = σ2
        σ2 = dot(ε, ε) / (N - m - Q)
        if (σ2 ≈ σ2prev) σ2 = σ2prev; break end
        μ, ϕ, θ, β = ls_params(ϕ0, c,  p, q)
    end
    return μ, ϕ, θ, σ2
end

hannan_rissanen(z::AbstractVector, c::Bool, p::Integer, q::Integer; m::Integer=20, n::Integer=1) = hannan_rissanen(z, c, collect(1:p), collect(1:q); m=m, n=n)
hannan_rissanen(z::AbstractVector, p::Integer, q::Integer; m::Integer=20, n::Integer=1) = hannan_rissanen(z, true, p, q; m=m, n=n)

function forecast(model::M, z::AbstractVector{T}) where {p,q,T,M <: ARMAModel{p,q,T}}
    N = length(z)
    a = Vector{T}(undef, N + 1)
    a[1] = zero(T)
    for i in 1:N
        a[i + 1] = z[i] - model.μ
        for j in 1:min(q, i)
            a[i + 1] += model.θ[j] * a[i - j + 1]
        end
        for j in 1:min(p, i - 1)
            a[i + 1] -= model.ϕ[j] * z[i - j]
        end
    end
    zhat = model.μ
    for j in 1:min(N, q)
        zhat -= model.θ[j] * a[N - j + 2]
    end
    for j in 1:min(N, p)
        zhat += model.ϕ[j] * z[N - j + 1]
    end
    return zhat
end

function MA∞(model::M, m::Integer) where {p,q,T,M <: ARMAModel{p,q,T}}
    ψ = Vector{T}(undef, m)
    for j in 1:m
        ψ[j] = - (j > q ? zero(T) : model.θ[j])
        for k in 1:p
            ψ[j] += model.ϕ[k] * (j - k == 0 ? one(T) : j - k < 0 ? zero(T) : ψ[j - k])
        end
    end
    return MAModel{m}(model.μ / (1 - sum(model.ϕ)), .-ψ, model.σ2)
end

function AR∞(model::M, m::Integer) where {p,q,T,M <: ARMAModel{p,q,T}}
    π = Vector{T}(undef, m)
    for j in 1:m
        π[j] = (j > p ? zero(T) : model.ϕ[j])
        for k in 1:q
            π[j] += model.θ[k] * (j - k == 0 ? -one(T) : j - k < 0 ? zero(T) : π[j - k])
        end
    end
    μ = model.μ / (1 - sum(model.ϕ)) * (1 - sum(π))
    μ = isnan(μ) ? zero(T) : μ
    return ARModel{m}(μ, π, model.σ2)
end