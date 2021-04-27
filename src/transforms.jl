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

function difference(z::AbstractVector; d::Integer=1,k::Integer=1)
    N = length(z)
    differenced = copy(z)
    @inbounds for j in 1:d
        for i in 1:N - j * k
            differenced[N - i + 1] -= differenced[N - i + 1 - k]
        end
    end
    return differenced[1 + d * k:N]
end

function integrate(z::AbstractVector{T}, z0::AbstractVector; d::Integer=1,k::Integer=1) where T
    N = length(z)
    integrated = Vector{T}(undef, N + d * k)
    N0 = length(z0)
    for i in 1:N + d * k
        integrated[i] = i > d * k ? z[i - d * k] : i > N0 ? zero(T) : z0[i]
    end
    for j in 1:d
        for i in (1 + k):N + d * k
            integrated[i] += integrated[i - k]
        end    
    end
    return integrated
end

integrate(z::AbstractVector{T}; d::Integer=1,k::Integer=1) where T = integrate(z, T[], d=d, k=k)