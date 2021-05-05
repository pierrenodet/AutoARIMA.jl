abstract type AbstractModel{T} end

abstract type AbstractParams end

function k(m::M) where {M <: AbstractModel}
    return sum(m.ϕ .!= 0) + sum(m.θ .!= 0) + m.μ !=0 + 1
end

function fit(params::AbstractVector{AbstractParams}, z::AbstractVector; criterium=aicc)
    criteria = []
    models = []
    for param in params
        model = fit(param, z)
        perf = criterium(model, residuals(model, z))
        push!(criteria, perf)
        push!(models, model)
        println("param : $param")
        println("criterium : $perf")
    end
    return models[findmin(criteria)[2]]
end

forecast(model::M) where {T,M <: AbstractModel{T}} = forecast(model, T[])

function residuals!(α::AbstractVector{T}, model::M, z::AbstractVector{T}) where {T,M <: AbstractModel{T}}
    length(α) == length(z) || throw(DimensionMismatch("α should have the same length as z"))
    N = length(z)
    α[1] = z[1] - forecast(model)
    for i in 2:N
        α[i] = z[i] - forecast(model, view(z, 1:i - 1))
    end
    return α
end

residuals(model::M, z::AbstractVector{T}) where {T,M <: AbstractModel{T}} = residuals!(Vector{T}(undef, length(z)), model, z)