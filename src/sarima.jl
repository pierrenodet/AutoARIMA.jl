using StaticArrays

struct SARIMA{p,d,q,P,D,Q,s,T} 
    μ::T
    ϕ::SVector{p,T}
    θ::SVector{q,T}
    Φ::SVector{P,T}
    Θ::SVector{Q,T}
    σ2::T
end
    
const WN{T} = SARIMA{0,0,0,0,0,0,0,T}
const AR{p,T} = SARIMA{p,0,0,0,0,0,0,T}
const MA{q,T} = SARIMA{0,0,q,0,0,0,0,T}
const ARMA{p,q,T} = SARIMA{p,0,q,0,0,0,0,T}
const ARIMA{p,d,q,T} = SARIMA{p,d,q,0,0,0,0,T}

WN(μ::T,σ2::T) where T = SARIMA{0,0,0,0,0,0,0,T}(μ, SA{T}[], SA{T}[], SA{T}[], SA{T}[], σ2)

AR(μ::T,ϕ::SVector{p,T},σ2::T) where {p,T} = SARIMA{p,0,0,0,0,0,0,T}(μ, ϕ, SA{T}[], SA{T}[], SA{T}[], σ2)
AR{p}(μ::T,ϕ::AbstractVector{T},σ2::T) where {p,T} = SARIMA{p,0,0,0,0,0,0,T}(μ, SVector{p,T}(ϕ), SA{T}[], SA{T}[], SA{T}[], σ2)

MA(μ::T,θ::SVector{q,T},σ2::T) where {q,T} = SARIMA{0,0,q,0,0,0,0,T}(μ, SA{T}[], θ, SA{T}[], SA{T}[], σ2)
MA{q}(μ::T,θ::AbstractVector{T},σ2::T) where {q,T} = SARIMA{p,0,0,0,0,0,0,T}(μ, SA{T}[], SVector{q,T}(θ), SA{T}[], SA{T}[], σ2)

ARMA(μ::T,ϕ::SVector{p,T},θ::SVector{q,T},σ2::T) where {p,q,T} = SARIMA{p,0,q,0,0,0,0,T}(μ, ϕ, θ, SA{T}[], SA{T}[], σ2)
ARMA{p,q}(μ::T,ϕ::AbstractVector{T},θ::AbstractVector{T},σ2::T) where {p,q,T} = SARIMA{p,0,q,0,0,0,0,T}(μ, SVector{p,T}(ϕ), SVector{q,T}(θ), SA{T}[], SA{T}[], σ2)

ARIMA{d}(μ::T,ϕ::SVector{p,T},θ::SVector{q,T},σ2::T) where {p,d,q,T} = SARIMA{p,d,q,0,0,0,0,T}(μ, ϕ, θ, SA{T}[], SA{T}[], σ2)
ARIMA{p,d,q}(μ::T,ϕ::AbstractVector{T},θ::AbstractVector{T},σ2::T) where {p,d,q,T} = SARIMA{p,d,q,0,0,0,0,T}(μ, SVector{p,T}(ϕ), SVector{q,T}(θ), SA{T}[], SA{T}[], σ2)

SARIMA{d,D,s}(μ::T, ϕ::SVector{p,T}, θ::SVector{q,T}, Φ::SVector{P,T}, Θ::SVector{Q,T}, σ2::T) where {p,d,q,P,D,Q,s,T} = SARIMA{p,d,q,P,D,Q,s,T}(μ, ϕ, θ, Φ, Θ, σ2)
SARIMA{p,d,q,P,D,Q,s}(μ::T,ϕ::AbstractVector{T},θ::AbstractVector{T}, Φ::AbstractVector{T}, Θ::AbstractVector{T},σ2::T) where {p,d,q,P,D,Q,s,T} = SARIMA{p,d,q,P,D,Q,s,T}(μ, SVector{p,T}(ϕ), SVector{q,T}(θ), SVector{P,T}(Φ), SVector{Q,T}(Θ), σ2)

function k(::M) where {p,d,q,P,D,Q,s,T,M <: SARIMA{p,d,q,P,D,Q,s,T}}
    return p + q + P + Q + 2
end

function fit(::Type{SARIMA{p,d,q,P,D,Q,s}}, z::AbstractVector{T}) where {p,d,q,P,D,Q,s,T}
    N = length(z)
    N > d + D * s || throw(DomainError("SARIMA d D s requires at least d + D*s data points"))
    ∇sdz = difference(z, d=D, s=s)
    Μ, Φ, Θ, Σ2 = hannan_rissanen(∇sdz, P, Q)
    μ, ϕ, θ, σ2 = hannan_rissanen(difference(∇sdz, d=d), p, q)
    return SARIMA{d,D,s}(μ, SVector{p,T}(ϕ), SVector{q,T}(θ), SVector{P,T}(Φ), SVector{Q,T}(Θ), σ2)
end

function forecast(model::M, z::AbstractVector{T}) where {p,d,q,P,D,Q,s,T,M <: SARIMA{p,d,q,P,D,Q,s,T}}
    N = length(z)
    arma = ARMA(model.μ, model.ϕ, model.θ, model.σ2)
    sarma = ARMA(model.μ, model.Φ, model.Θ, model.σ2)
    ∇s = difference(z, d=D, s=s)
    ∇shat = forecast(sarma, ∇s)
    push!(∇s, ∇shat)
    i∇s = integrate(∇s, z, d=D, s=s)
    ∇z = difference(i∇s, d=d)
    ∇zhat = forecast(arma, ∇z)
    push!(∇z, ∇zhat)
    return last(integrate(∇z, z, d=d))
end

forecast(model::M) where {p,d,q,P,D,Q,s,T,M <: SARIMA{p,d,q,P,D,Q,s,T}} = forecast(model, T[])

struct SARIMASimulator{p,d,q,P,D,Q,s,T}
    model::SARIMA{p,d,q,P,D,Q,s,T}
    history::Vector{T}
end

struct SARIMASimulatorState{T}
    z::Vector{T}
end

function simulate(model::SARIMA{p,d,q,P,D,Q,s,T}, data::AbstractVector{T}) where {p,d,q,P,D,Q,s,T} 
    return SARIMASimulator(model, data)
end

simulate(model::SARIMA{p,d,q,P,D,Q,s,T}) where {p,d,q,P,D,Q,s,T} = SARIMASimulator(model, T[])

function Base.iterate(simulator::SARIMASimulator{p,d,q,P,D,Q,s,T}) where {p,d,q,P,D,Q,s,T}
    z = forecast(simulator.model, simulator.history)
    a = √simulator.model.σ2 * randn(T)
    push!(simulator.history, z + a)
    return (z + a, SARIMASimulatorState(simulator.history))
end

function Base.iterate(simulator::SARIMASimulator{p,d,q,P,D,Q,s,T}, state::SARIMASimulatorState{T}) where {p,d,q,P,D,Q,s,T}
    z = forecast(simulator.model, state.z)
    a = √simulator.model.σ2 * randn(T)
    push!(state.z, z + a)
    return (z + a, state)
end