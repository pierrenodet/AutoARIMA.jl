@recipe function f(::Type{Val{:tsaplot}}, x, y, z)

    N = length(x)

    grid := false
    legend := false
    linewidth --> 1

    @series begin
        seriestype := :stem
        seriescolor --> :black
        x := x
        y := y
    end

    bound = 1.96 / √N
    linestyle := :dash
    linecolor --> :blue

    @series begin
        seriestype := :hline
        y := -bound
    end

    @series begin
        seriestype := :hline
        y := bound
    end

    @series begin
        seriestype := :hline
        linestyle := :solid
        linecolor := :black
        linewidth := 1
        y := 0.0
    end

end

@userplot Correlogram
@recipe function f(c::Correlogram)
    if length(c.args) != 2 || !(typeof(c.args[1]) <: AbstractVector) ||
        !(typeof(c.args[2]) <: Integer)
        error("Correlograms should be given a vector and an integer.  Got: $(typeof(h.args))")
    end
    z, k = c.args
    seriestype := :tsaplot
    1:k, map(i -> autocorrelation(z, i), 1:k)
end

@userplot Partial_Correlogram
@recipe function f(c::Partial_Correlogram; recursive=true)
    if length(c.args) != 2 || !(typeof(c.args[1]) <: AbstractVector) ||
        !(typeof(c.args[2]) <: Integer)
        error("Correlograms should be given a vector and an integer.  Got: $(typeof(h.args))")
    end
    z, k = c.args
    seriestype := :tsaplot

    if recursive
        ρ = map(i -> autocorrelation(z, i), 1:k)
        T = typeof(zero(eltype(z)) / 1)
        ϕ = zeros(T, k)
        ϕpp = zeros(T, k)
        σ2 = Ref{T}(autocovariance(z, 0))
        for i in 1:k
            levinson_durbin!(ϕ, σ2, ρ, i)
            ϕpp[i] = ϕ[i]
        end
    else
        ϕpp = map(i -> partial_autocorrelation(z, i), 1:k)
    end

    1:k, ϕpp
end

