using Test, AutoARIMA, Statistics

@testset "TimeSeriesAnalysisForecastingandControl" begin
    @testset "Table6.5" begin
        μ, ϕ, θ, σ2 = hannan_rissanen(seriesA, 1, 1)
        @test μ ≈ 2.45 rtol = 0.2
        @test ϕ ≈ [0.87] rtol = 0.05
        @test θ ≈ [-0.48] rtol = 0.05
        @test σ2 ≈ 0.098 rtol = 0.05

        θ, σ2 = innovations(diff(seriesA), 1, m=3)
        @test θ ≈ [-0.53] rtol = 0.05
        @test σ2 ≈ 0.107 rtol = 0.05

        θ, σ2 = innovations(diff(seriesB), 1, m=1)
        @test θ ≈ [0.09] rtol = 0.05
        @test σ2 ≈ 52.2 rtol = 0.05

        μ, ϕ, σ2 = levinson_durbin(diff(seriesC), 1)
        @test μ ≈ 0.0 atol = 0.01
        @test ϕ ≈ [0.81] rtol = 0.05
        @test σ2 ≈ 0.019 rtol = 0.05

        θ, σ2 = innovations(diff(diff(seriesC)), 2, m=2)
        @test θ ≈ [-0.09,-0.07] rtol = 0.2
        @test σ2 ≈ 0.020 rtol = 0.05

        μ, ϕ, σ2 = levinson_durbin(seriesD, 1)
        @test μ ≈ 1.32 rtol = 0.05
        @test ϕ ≈ [0.86] rtol = 0.05
        @test σ2 ≈ 0.093 rtol = 0.05

        θ, σ2 = innovations(diff(seriesD), 1, m=1)
        @test θ ≈ [-0.05] rtol = 0.05
        @test σ2 ≈ 0.096 rtol = 0.05

        μ, ϕ, σ2 = levinson_durbin(seriesE, 2)
        @test μ ≈ 14.9 rtol = 0.05
        @test ϕ ≈ [1.32,-0.63] rtol = 0.05
        @test σ2 ≈ 289.0 rtol = 0.05

        μ, ϕ, σ2 = levinson_durbin(seriesE, 3)
        @test μ ≈ 13.7 rtol = 0.05
        @test ϕ ≈ [1.37,-0.74,0.08] rtol = 0.05
        @test σ2 ≈ 287.0 rtol = 0.05

        μ, ϕ, σ2 = levinson_durbin(seriesF, 2)
        @test μ ≈ 58.3 rtol = 0.05
        @test ϕ ≈ [-0.32,0.18] rtol = 0.05 
        @test σ2 ≈ 115.0 rtol = 0.05
    end

    @testset "Table3.1" begin
        ϕ = map(i -> partial_autocorrelation(seriesF, i), 1:15)
        @test ϕ ≈ [-0.39,0.18,0.00,-0.04,-0.07,-0.12,0.02,0.00,-0.06,0.00,0.14,-0.01,0.09,0.17,0.00] rtol = 0.05
    end

    @testset "Table2.1" begin
        ϕ = map(i -> autocorrelation(seriesF, i), 1:15)
        @test ϕ ≈ [-0.39,0.30,-0.17,0.07,-0.10,-0.05,0.04,-0.04,0.00,0.01,0.11,-0.07,0.15,0.04,0.01] rtol = 0.05
    end

    # @testset "Table4.2" begin
    #     @test AR(ARMA(1.0,SA[-0.3],SA[-0.5],1.0),7).ϕ ≈ SA[0.2,0.4,0.2,0.1,0.05,0.025,0.0125]
    # end

end


