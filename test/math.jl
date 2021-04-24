using Test, AutoARIMA

@testset "MathematicalProperties" begin

    @test partial_correlogram(seriesF, 2)[2] ≈ map(i -> partial_autocorrelation(seriesF, i), 1:2)
    @test correlogram(seriesF, 2)[2] ≈ map(i -> autocorrelation(seriesF, i), 0:2)
    @test reduce(&, yule_walker(seriesF, 15) .≈ levinson_durbin(seriesF, 15))

end