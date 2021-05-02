struct Simulator{T}
    model::AbstractModel{T}
    z::Vector{T}
    x::Vector{T}
end

struct SimulatorState{T}
    z::Vector{T}
    x::Vector{T}
end

function simulate(model::AbstractModel{T},  z::AbstractVector{T},  x::AbstractVector{T}) where T 
    return Simulator(model,  z,  x)
end

simulate(model::AbstractModel{T},  z::AbstractVector{T}) where T = Simulator(model,  z,  T[])
simulate(model::AbstractModel{T}) where T = Simulator(model, T[], T[])

function Base.iterate(simulator::Simulator{T}) where T
    z = forecast(simulator.model, simulator.z)
    a = √simulator.model.σ2 * randn(T)
    push!(simulator.z, z + a)
    return (z + a, SimulatorState(simulator.z,simulator.x))
end

function Base.iterate(simulator::Simulator{T}, state::SimulatorState{T}) where T
    z = forecast(simulator.model, state.z)
    a = √simulator.model.σ2 * randn(T)
    push!(state.z, z + a)
    return (z + a, state)
end