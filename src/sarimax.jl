using StaticArrays

struct SARIMAX{p,d,q,P,D,Q,s,T} 
    μ::T
    ϕ::SVector{p,T}
    θ::SVector{q,T}
    σ2::T
end
    
const WN{T} = SARIMAX{0,0,0,0,0,0,0,T}
const AR{p,T} = SARIMAX{p,0,0,0,0,0,0,T}
const MA{q,T} = SARIMAX{0,0,q,0,0,0,0,T}
const ARMA{p,q,T} = SARIMAX{p,0,q,0,0,0,0,T}
const ARIMA{p,d,q,T} = SARIMAX{p,d,q,0,0,0,0,T}

WN(μ::T,σ2::T) where T = SARIMAX{0,0,0,0,0,0,0,T}(μ, SA{T}[], SA{T}[], σ2)
AR(μ::T,ϕ::SVector{p,T},σ2::T) where {p,T} = SARIMAX{p,0,0,0,0,0,0,T}(μ, ϕ, SA{T}[], σ2)
MA(μ::T,θ::SVector{q,T},σ2::T) where {q,T} = SARIMAX{0,0,q,0,0,0,0,T}(μ, SA{T}[], θ, σ2)
ARMA(μ::T,ϕ::SVector{p,T},θ::SVector{q,T},σ2::T) where {p,q,T} = SARIMAX{p,0,q,0,0,0,0,T}(μ, ϕ, θ, σ2)

struct SARIMAXSimulator{p,d,q,P,D,Q,s,T}
    model::SARIMAX{p,d,q,P,D,Q,s,T}
    history::Vector{T}
end

struct SARIMAXSimulatorState{T}
    z::Vector{T}
end

function simulate(model::SARIMAX{p,d,q,P,D,Q,s,T}, data::AbstractVector{T}) where {p,d,q,P,D,Q,s,T} 
    return SARIMAXSimulator(model, data)
end

simulate(model::SARIMAX{p,d,q,P,D,Q,s,T}) where {p,d,q,P,D,Q,s,T} = SARIMAXSimulator(model, T[])

function Base.iterate(simulator::SARIMAXSimulator{p,d,q,P,D,Q,s,T}) where {p,d,q,P,D,Q,s,T}
    z = forecast(simulator.model, simulator.history)
    a = √simulator.model.σ2 * randn(T)
    push!(simulator.history, z + a)
    return (z + a, SARIMAXSimulatorState(simulator.history))
end

function Base.iterate(simulator::SARIMAXSimulator{p,d,q,P,D,Q,s,T}, state::SARIMAXSimulatorState{T}) where {p,d,q,P,D,Q,s,T}
    z = forecast(simulator.model, state.z)
    a = √simulator.model.σ2 * randn(T)
    push!(state.z, z + a)
    return (z + a, state)
end