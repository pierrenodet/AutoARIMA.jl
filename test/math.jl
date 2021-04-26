using Test, AutoARIMA

@testset "MathematicalProperties" begin

    @test partial_correlogram(seriesF, 2)[2] ≈ map(i -> partial_autocorrelation(seriesF, i), 1:2)
    @test correlogram(seriesF, 2)[2] ≈ map(i -> autocorrelation(seriesF, i), 0:2)
    @test reduce(&, yule_walker(seriesF, 15) .≈ levinson_durbin(seriesF, 15))
    
    @testset "MA(q) forecast after q steps should be equal to the constant" begin
        μ = 0.1
        ma = MA(μ,SA[0.2,0.3,-0.5],1.0)
        z = Float64[0.2,0.5,0.1]
        f = Float64[]
        for i in 1:10
            zhat = forecast(ma,z)
            push!(z,zhat)
            push!(f,zhat)
        end
        @test all(f[4:end] .≈ μ)
    end

end