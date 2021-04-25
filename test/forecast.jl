using Test, AutoARIMA, StaticArrays

@testset "Forecast" begin

    @test forecast(MA(0.0, SA[0.7], 1.0), SA[3.0,8,2,5,6]) ≈ 1.01941
    
    @test forecast(AR(1.0, SA[0.5, 0.5], 1.0), SA[3.0,2]) ≈ 3.5

end