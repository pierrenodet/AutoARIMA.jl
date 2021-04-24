using StaticArrays, LinearAlgebra

function hannan_rissanen(z::AbstractVector, p::Integer, q::Integer)
    T = typeof(zero(eltype(z)) / 1)
    N = length(z)
    m = 20 + max(p, q)
    μinf, ϕinf, σ2inf = levinson_durbin(z, m)
    ar = AR(μinf, SVector{m}(ϕinf), σ2inf)
    a = Vector{T}(undef, N - m)
	for i in 1:(N - m)
		a[i] = z[i + m] - forecast(ar, view(z, 1:i + m - 1))
	end
    Z = Matrix{T}(undef, N - m - q, p + q + 1)
    Z[:,1] .= one(T)
    for j in 1:p
        Z[:,j + 1] = view(z, j + m:N + j - q - 1)
    end
    for j in 1:q
        Z[:,j + p + 1] = view(a, j:N - m + j - q - 1)
    end
    ϕ = Z \ view(z, m + q + 1:N)
    ε = view(z, m + q + 1:N) - Z * ϕ
    σ2 = dot(ε, ε) / (N - m - q)
    μ = ϕ[1]
    θ = reverse(view(ϕ, p + 2:q + p + 1))
    ϕ = reverse(view(ϕ, 2:p + 1))
    return μ, ϕ, θ, σ2
end

# function MA(model::M, m::Integer) where {M <: ARMA{p,q,T}}
#     T = typeof(zero(eltype(z)) / 1)
#     ψ = Vector{T}(undef, m)
#     for j in 1:m
#         tmp = zero(T)
#         for k in 1:max(j, p)
#             tmp += model.ϕ[k] * ψ[j - k + 1]
#         end
#         ψ[j] = model.θ[j] + tmp
#     end
#     return MA(model.μ, SVector{m}(ψ), model.σ2)    
# end