function ls_params(ϕ::AbstractVector{T}, c::Bool, p::AbstractVector{<:Integer}, q::AbstractVector{<:Integer}) where T
    P = isempty(p) ? 0 : maximum(p)
    Np = length(p)
    Q = isempty(q) ? 0 : maximum(q)
    Nq = length(q)
    ϕ0 = zeros(T, P)
    θ0 = zeros(T, Q)
    μ0 = c ? ϕ[1] : zero(T)
    for (i, l) in enumerate(p)
        ϕ0[l] = ϕ[i + c]
    end
    for (i, l) in enumerate(q) 
        θ0[l] = ϕ[i + Np + c]
    end
    β0 = ϕ[Nq + Np + 1 + c:end]
    return μ0, ϕ0, θ0, β0
end

function ls_matrix!(Z::AbstractMatrix{T}, z::AbstractVector{T}, a::AbstractVector{T}, x::AbstractMatrix{T}, c::Bool, p::AbstractVector{<:Integer}, q::AbstractVector{<:Integer}) where T
    all(q .!= 0) || throw(ArgumentError("order of ma can't be zero"))
    all(p .!= 0) || throw(ArgumentError("order of ar can't be zero"))
    N = length(z)
    P = isempty(p) ? 0 : maximum(p)
    Np = length(p)
    Q = isempty(q) ? 0 : maximum(q)
    Nq = length(q)
    X = size(x, 2)
    if c
        Z[:,1] = ones(T, N - P - Q)
    end
    for (i, l) in enumerate(p)
        Z[:,i + c] = view(z, P + Q - l + 1:N - l)
    end
    for (i, l) in enumerate(q)
        Z[:,i + Np + c] = .-view(a, P + Q - l + 1:N - l)
    end
    if X > 0 Z[:, Np + Nq + 1 + c:end] = view(x, P:N - 1, :) end
    return Z, z[P + Q + 1:N]
end

function ls_matrix(z::AbstractVector{T}, a::AbstractVector{T}, x::AbstractMatrix{T}, c::Bool, p::AbstractVector{<:Integer}, q::AbstractVector{<:Integer}) where T
    N = length(z)
    P = isempty(p) ? 0 : maximum(p)
    Np = length(p)
    Q = isempty(q) ? 0 : maximum(q)
    Nq = length(q)
    X = size(x, 2)
    Z = Matrix{T}(undef, N - P - Q, Np + Nq + X + c)
    ls_matrix!(Z, z, a, x, c, p, q)
end

function least_squares(z::AbstractVector{T}, a::AbstractVector{T}, x::AbstractMatrix{T}, c::Bool, p::AbstractVector{<:Integer}, q::AbstractVector{<:Integer}) where T
    Z, z = ls_matrix(z, a, x, c, p, q)
    ϕ = Z \ z
    ε = z - Z * ϕ
    σ2 = dot(ε, ε) / length(ε)
    μ, ϕ, θ, β = ls_params(ϕ, c, p, q)
    return μ, ϕ, θ, β, σ2
end

least_squares(z::AbstractVector{T}, a::AbstractVector{T}, x::AbstractMatrix{T}, c::Bool, p::Integer, q::Integer) where T = least_squares(z, a, x, collect(0:p), collect(1:q))