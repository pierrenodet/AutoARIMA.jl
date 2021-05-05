struct Simulator{T}
    model::AbstractModel{T}
    z::Vector{T}
    x::Vector{T}
end

struct SimulatorState{T}
    z::Vector{T}
    a::Vector{T}
end

function simulate(model::AbstractModel{T},  z::AbstractVector{T},  x::AbstractVector{T}) where T 
    return Simulator(model,  z,  x)
end

simulate(model::AbstractModel{T},  z::AbstractVector{T}) where T = Simulator(model,  z,  T[])
simulate(model::AbstractModel{T}) where T = Simulator(model, T[], T[])

function Base.iterate(simulator::Simulator{T}) where T
    α = residuals(simulator.model, simulator.z)
    z = forecast(simulator.model, simulator.z, α)
    a = √simulator.model.σ2 * randn(T)
    push!(simulator.z, z + a)
    push!(α, a)
    return (z + a, SimulatorState(simulator.z, α))
end

function Base.iterate(simulator::Simulator{T}, state::SimulatorState{T}) where T
    z = forecast(simulator.model, state.z, state.a)
    a = √simulator.model.σ2 * randn(T)
    push!(state.z, z + a)
    push!(state.a, a)
    return (z + a, state)
end