using Statistics, Optim

function boxcox(z, λ)
    reduce(&, z .> 0) || throw(DomainError("boxcox requires strictly positive data"))
	if λ == 0
		return log.(z) 
	else
		return (z.^λ .- 1) ./ λ
	end
end

function guerrero(z; lower=-1, upper=2, s=2)
    s >= 2 || throw(ArgumentError("guerrero requires seasonality to be higher or equal to 2"))
    N = length(z)
    rz = reshape(view(z, (N % s + 1):N), N ÷ s, s)
    m = mean(rz, dims=1)
    sd = std(rz, mean=m, dims=1)
    function cv(λ)
        w = sd ./ (m.^(1 - λ))
        mw = mean(w)
        return std(w, mean=mw) / mw
    end
    return optimize(cv, lower, upper).minimizer
end

function difference(z::AbstractVector{T}; d::Integer=1,s::Integer=1) where T
    N = length(z)
    differenced = Vector{T}(undef, N)
    @inbounds @simd for i in 1:N
        differenced[i] = z[i]
    end
    @inbounds for j in 1:d
        for i in 1:N - j * s
            differenced[N - i + 1] -= differenced[N - i + 1 - s]
        end
    end
    return differenced[1 + d * s:N]
end

function integrate(z::AbstractVector{T}, z0::AbstractVector; d::Integer=1,s::Integer=1) where T
    N = length(z)
    integrated = Vector{T}(undef, N + d * s)
    N0 = length(z0)
    for i in 1:N + d * s
        integrated[i] = i > d * s ? z[i - d * s] : i > N0 ? zero(T) : z0[i]
    end
    for j in 1:d
        for i in (1 + s):N + d * s
            integrated[i] += integrated[i - s]
        end    
    end
    return integrated
end

integrate(z::AbstractVector{T}; d::Integer=1,s::Integer=1) where T = integrate(z, T[], d=d, s=s)