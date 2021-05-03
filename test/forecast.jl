using Test, AutoARIMA

@testset "Forecast" begin

    @test forecast(MAModel{1}(0.0, [-0.7], 1.0), [3.0,8,2,5,6]) ≈ 1.01941
    
    @test forecast(ARModel{2}(1.0, [0.5, 0.5], 1.0), [3.0,2]) ≈ 3.5

    @test forecast(ARModel{2}(10.0, [0.5, 0.5], 1.0), [3.0,2]) ≈ 12.5

end