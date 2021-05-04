using Test, AutoARIMA, Statistics

@testset "Time Series Analysis Forecasting and Control" begin
    @testset "Table6.5" begin
        arma = fit(ARMAParams(true, 1,1), seriesA)
        @test arma.μ ≈ 2.45 rtol = 0.2
        @test arma.ϕ ≈ [0.87] rtol = 0.05
        @test arma.θ ≈ [0.48] rtol = 0.05
        @test arma.σ2 ≈ 0.098 rtol = 0.05

        θ, σ2 = innovations(diff(seriesA), 1, n=10)
        @test θ ≈ [0.53] rtol = 0.01
        @test σ2 ≈ 0.107 rtol = 0.01

        θ, σ2 = innovations(diff(seriesB), 1)
        @test θ ≈ [-0.09] rtol = 0.05
        @test σ2 ≈ 52.2 rtol = 0.05

        ar = fit(ARParams(false, 1), diff(seriesC))
        @test ar.μ ≈ 0.0 atol = 0.01
        @test ar.ϕ ≈ [0.81] rtol = 0.05
        @test ar.σ2 ≈ 0.019 rtol = 0.05

        θ, σ2 = innovations(diff(diff(seriesC)), 2, n=30)
        @test θ ≈ [0.09,0.07] rtol = 0.05
        @test σ2 ≈ 0.020 rtol = 0.05

        ar = fit(ARParams(1), seriesD)
        @test ar.μ ≈ 1.32 rtol = 0.05
        @test ar.ϕ ≈ [0.86] rtol = 0.05
        @test ar.σ2 ≈ 0.093 rtol = 0.05

        θ, σ2 = innovations(diff(seriesD), 1)
        @test θ ≈ [0.05] rtol = 0.05
        @test σ2 ≈ 0.096 rtol = 0.01

        ar = fit(ARParams(2), seriesE)
        @test ar.μ ≈ 14.9 rtol = 0.05
        @test ar.ϕ ≈ [1.32,-0.63] rtol = 0.05
        @test ar.σ2 ≈ 289.0 rtol = 0.05

        ar = fit(ARParams(3), seriesE)
        @test ar.μ ≈ 13.7 rtol = 0.05
        @test ar.ϕ ≈ [1.37,-0.74,0.08] rtol = 0.05
        @test ar.σ2 ≈ 287.0 rtol = 0.05

        ar = fit(ARParams(2), seriesF)
        @test ar.μ ≈ 58.3 rtol = 0.05
        @test ar.ϕ ≈ [-0.32,0.18] rtol = 0.05 
        @test ar.σ2 ≈ 115.0 rtol = 0.05
    end

    @testset "Table3.1" begin
        ϕ = map(i -> partial_autocorrelation(seriesF, i), 1:15)
        @test ϕ ≈ [-0.39,0.18,0.00,-0.04,-0.07,-0.12,0.02,0.00,-0.06,0.00,0.14,-0.01,0.09,0.17,0.00] rtol = 0.05
    end

    @testset "Table2.1" begin
        ϕ = map(i -> autocorrelation(seriesF, i), 1:15)
        @test ϕ ≈ [-0.39,0.30,-0.17,0.07,-0.10,-0.05,0.04,-0.04,0.00,0.01,0.11,-0.07,0.15,0.04,0.01] rtol = 0.05
    end

    @testset "Section9.1" begin
        sarima = fit(MSARIMAParams([0,0], [1,1], [1,1], [1,12]), log.(seriesG))
        @test sarima.ϕ ≈ [1,0,0,0,0,0,0,0,0,0,0,1,-1] rtol = 0.05
        # @test sarima.θ[1] ≈ 0.4 rtol = 0.05
        # @test sarima.θ[12] ≈ 0.6 rtol = 0.05
        @test sarima.θ[13] ≈ -sarima.θ[1] * sarima.θ[12] rtol = 0.05
    end

    # @testset "Section10.1" begin
    #     arma = MA∞(fit(ARMAParams(1, 1), seriesA), 30)
    #     @test arma.θ ≈ [0.34,0.32,0.29,0.265,0.24,0.22,0.21,0.19,0.17,0.16,0.15,0.14,0.12,0.11,0.10,0.10,0.09,0.082,0.075,0.07,0.065,0.06,0.054,0.05,0.046,0.043,0.039,0.36,0.033,0.030] rtol = 0.05
    #     arima = MA∞(fit(ARIMAParams(0, 1, 1), seriesA), 30)
    #     @test arima.θ ≈ fill(0.29, 30)
    # end

    @testset "Table4.2" begin
        @test AR∞(toarma(ARIMAParams(1, 1, 1), ARMAModel{1,1}(1.0, [-0.3], [0.5], 1.0)), 7).ϕ ≈ [0.2,0.4,0.2,0.1,0.05,0.025,0.0125]
    end

end


