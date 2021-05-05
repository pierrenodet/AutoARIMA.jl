function finddifference(z, p, s, dmax)
    Δz = z
    d = 0
    while d < dmax && pvalue(ADFTest(Δz, :none, p)) > 0.05 d += 1; Δz = difference(z, d=d, s=s)  end
    return d
end

function boxjenkins(z, c::Bool, pmax::Integer, dmax::Integer, qmax::Integer, s::AbstractVector{<:Integer}; criterium=aicc)
    λ = guerrero(z)
    λ = λ < 0.2 ? 0.0 : λ
    z = boxcox(z, λ)
    d = [finddifference(z, pmax, s, dmax) for s in s]
    params = AbstractParams[]
    S = length(s)
    for i in 0:pmax, j in 0:qmax
        push!(params, MSARIMAParams(c, fill(i, S), d, fill(j, S), s))
    end
    return fit(params, z, criterium=criterium)
end