struct MSARIMAParams{T <: Integer} <: AbstractParams
    c::Vector{Bool}
    p::Vector{Vector{T}}
    d::Vector{T}
    q::Vector{Vector{T}}
    s::Vector{T}
    function MSARIMAParams(c::Vector{Bool}, p::Vector{Vector{T}}, d::Vector{T}, q::Vector{Vector{T}}, s::Vector{T}) where T <: Integer
        length(c) == length(p) == length(q) == length(d) == length(s) || throw(DimensionMismatch("c,p,q,d and s should have the same length"))
        return new{T}(c, p, d, q, s)
    end
end

MSARIMAParams(c::Vector{Bool}, p::Vector{T},d::Vector{T},q::Vector{T},s::Vector{T}) where T <: Integer = MSARIMAParams(c, map(i -> collect(1:i), p), d, map(i -> collect(1:i), q), s)
MSARIMAParams(p::Vector{T},d::Vector{T},q::Vector{T},s::Vector{T}) where T <: Integer = MSARIMAParams(fill(false, length(p)), p, d, q, s)

function toarma(params::MSARIMAParams, armas::Vector{ARMAModel})
    M = length(armas)
    ϕ0 = Polynomial(1)
    θ0 = Polynomial(1)
    for i in 1:M
        ϕ0 *= Polynomial([1;.-armas[i].ϕ]) * Polynomial(1 - variable()^params.s[i])^params.d[i]
        θ0 *= Polynomial([1;.-armas[i].θ])
    end
    p0 = length(ϕ0) - 1
    q0 = length(θ0) - 1
    return ARMAModel{p0,q0}(armas[1].μ, .-coeffs(ϕ0)[2:end], .-coeffs(θ0)[2:end], armas[1].σ2)
end

function fit(params::MSARIMAParams, z::AbstractVector; n::Integer=10)
    μ = mean(z)
    z = z .- μ
    M = length(params.p)
    N = length(z)
    t0 = maximum(params.d) * (1 + maximum(params.s))
    N > t0 || throw(DomainError("ARIMA requires at least d+D*s data points"))
    armas = ARMAModel[]
    α = zeros(N - t0)
    for i in 2:M
        P = isempty(params.p[i]) ? 0 : maximum(params.p[i] .* params.s[i])
        Q = isempty(params.q[i]) ? 0 : maximum(params.q[i] .* params.s[i])
        sarima = ARMAModel{P,Q}(hannan_rissanen(difference(z, d=params.d[i], s=params.s[i]), z, params.c[i], params.p[i] .* params.s[i], params.q[i] .* params.s[i], n=n)...)
        arma = toarma(MSARIMAParams([params.c[i]], [params.p[i]], [params.d[i]], [params.q[i]], [params.s[i]]), ARMAModel[sarima])
        push!(armas, sarima)
        println(sarima)
        println(arma)
        α .+= residuals(arma, z)[1+t0:end]
    end
    P = isempty(params.p[1]) ? 0 : maximum(params.p[1] .* params.s[1])
    Q = isempty(params.q[1]) ? 0 : maximum(params.q[1] .* params.s[1])
    arima = ARMAModel{P,Q}(hannan_rissanen(difference(α, d=params.d[1], s=params.s[1]), α, params.c[1], params.p[1] .* params.s[1], params.q[1] .* params.s[1], n=n)...)
    println(arima)
    pushfirst!(armas, arima)
    return toarma(params, armas), α
end

# function fit(params::MSARIMAParams, z::AbstractVector; n::Integer=10)
#     M = length(params.p)
#     N = length(z)
#     t0 = maximum(params.d) * (1 + maximum(params.s))
#     N > t0 || throw(DomainError("ARIMA requires at least d+D*s data points"))
#     armas = ARMAModel[]
#     S = maximum(params.s)
#     α = zeros(N)
#     for i in 2:M
#         sarima = fit(ARMAParams(params.c[i], params.p[i], params.q[i]), difference(z, d=params.d[i], s=params.s[i]), n=n)
#         arma = toarma(MSARIMAParams([params.c[i]], [params.p[i]], [params.d[i]], [params.q[i]], [params.s[i]]), ARMAModel[sarima])
#         push!(armas, sarima)
#         println(sarima)
#         println(arma)
#         α += residuals(arma, z)[1:end]
#     end
#     arma = fit(ARMAParams(params.c[1], params.p[1], params.q[1]), difference(z, d=params.d[1], s=params.s[1]), n=n)
#     pushfirst!(armas, arma)
#     return toarma(params, armas), α
# end

# using Plots, Statistics, StaticArrays, LinearAlgebra, Polynomials, BenchmarkTools, Optim

# T = 50
# M = 30
# λ = guerrero(seriesG)
# println(λ)
# z = log.(seriesG)
# sarima, α = fit(MSARIMAParams([false,false], [0,0], [1,1], [1,1], [1,12]), z[1:T])
# println(sarima)
# plot(z[M:end])
# plot!((z - residuals(sarima, z))[M:end])
# h = z[1:T]
# for i in T + 1:length(z) + 30
#     push!(h, forecast(sarima, h))
# end
# plot!(h[M:end])
