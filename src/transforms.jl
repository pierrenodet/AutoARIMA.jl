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