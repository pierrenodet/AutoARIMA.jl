function ls_params(ϕ::AbstractVector{T}, p::AbstractVector{<:Integer}, q::AbstractVector{<:Integer}) where T
    P = isempty(p) ? 0 : maximum(p)
    Np = length(p)
    Q = isempty(q) ? 0 : maximum(q)
    Nq = length(q)
    ϕ0 = zeros(T, P)
    θ0 = zeros(T, Q)
    μ0 = zero(T)
    for (i, l) in enumerate(p)
        if l == 0
            μ0 += ϕ[i]
        else    
            ϕ0[l] = ϕ[i]
        end
    end
    for (i, l) in enumerate(q) 
        θ0[l] = ϕ[i + Np]
    end
    β0 = ϕ[Nq + Np + 1:end]
    return μ0, ϕ0, θ0, β0
end

function ls_matrix!(Z::AbstractMatrix{T}, z::AbstractVector{T}, a::AbstractVector{T}, x::AbstractMatrix{T}, p::AbstractVector{<:Integer}, q::AbstractVector{<:Integer}) where T
    all(q .!= 0) || throw(ArgumentError("order of ma can't be zero"))
    sum(p .== 0) < 2 || throw(ArgumentError("order of ar can contain only one zero for constant term"))
    N = length(z)
    P = isempty(p) ? 0 : maximum(p)
    Np = length(p)
    Q = isempty(q) ? 0 : maximum(q)
    Nq = length(q)
    X = size(x, 2)
    for (i, l) in enumerate(p)
        if l == 0 
            Z[:,i] = ones(T, N - P - Q)
        else  
            Z[:,i] = view(z, P + Q - l + 1:N - l)
        end
    end
    for (i, l) in enumerate(q)
        Z[:,i + Np] = .-view(a, P + Q - l + 1:N - l)
    end
    if X > 0 Z[:, Np + Nq + 1:Np + Nq + X] = view(x, P:N - 1, :) end
    return Z, z[P + Q + 1:N]
end

function ls_matrix(z::AbstractVector{T}, a::AbstractVector{T}, x::AbstractMatrix{T}, p::AbstractVector{<:Integer}, q::AbstractVector{<:Integer}) where T
    N = length(z)
    P = isempty(p) ? 0 : maximum(p)
    Np = length(p)
    Q = isempty(q) ? 0 : maximum(q)
    Nq = length(q)
    X = size(x, 2)
    Z = Matrix{T}(undef, N - P - Q, Np + Nq + X)
    ls_matrix!(Z, z, a, x, p, q)
end

function least_squares(z::AbstractVector{T}, a::AbstractVector{T}, x::AbstractMatrix{T}, p::AbstractVector{<:Integer}, q::AbstractVector{<:Integer}) where T
    Z, z = ls_matrix(z, a, x, p, q)
    ϕ = Z \ z
    ε = z - Z * ϕ
    σ2 = dot(ε, ε) / length(ε)
    μ, ϕ, θ, β = ls_params(ϕ, p, q)
    return μ, ϕ, θ, β, σ2
end

least_squares(z::AbstractVector{T}, a::AbstractVector{T}, x::AbstractMatrix{T}, p::Integer, q::Integer) where T = least_squares(z, a, x, collect(0:p), collect(1:q))