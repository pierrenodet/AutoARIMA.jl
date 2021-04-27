using Test, AutoARIMA, StaticArrays, Base.Iterators

@testset "Simulated" begin

    @testset "AR" begin

        ar = AR(0.0, SA[0.8,-0.2], 1.0)
        z = Float64[]
        for s in take(simulate(ar), 10000)
            push!(z, s)
        end
        μ, ϕ, σ2 = levinson_durbin(z, 2)
        @test μ ≈ 0.0 atol = 0.05
        @test ϕ ≈ [0.8,-0.2] rtol = 0.05
        @test σ2 ≈ 1.0 rtol = 0.05

    end

    @testset "AR with HR" begin

        ar = AR(0.0, SA[0.8,-0.2], 1.0)
        z = Float64[]
        for s in take(simulate(ar), 50000)
            push!(z, s)
        end
        μ, ϕ, θ, σ2 = hannan_rissanen(z, 2, 0)
        @test μ ≈ 0.0 atol = 0.05
        @test ϕ ≈ [0.8,-0.2] rtol = 0.05
        @test σ2 ≈ 1.0 rtol = 0.05

    end

    @testset "MA with HR" begin

        ma = MA(3.0, SA[0.2,-0.2], 1.0)
        z = Float64[]
        for s in take(simulate(ma), 50000)
            push!(z, s)
        end
        μ, ϕ, θ, σ2 = hannan_rissanen(z, 0, 2)
        @test μ ≈ 3.0 rtol = 0.05
        @test θ ≈ [0.2,-0.2] rtol = 0.05
        @test σ2 ≈ 1.0 rtol = 0.05
        
    end

    @testset "ARMA" begin

        arma = ARMA(5.0, SA[-0.7,0.2], SA[0.7,0.2], 1.0)
        z = Float64[]
        for s in take(simulate(arma), 50000)
            push!(z, s)
        end
        μ, ϕ, θ, σ2 = hannan_rissanen(z,2, 2)
        @test μ ≈ 5.0 rtol = 0.05
        @test ϕ ≈ [-0.7,0.2] rtol = 0.05
        @test θ ≈ [0.7,0.2] rtol = 0.1
        @test σ2 ≈ 1.0 rtol = 0.05
        
    end

    @testset "ARMA with MA Inf" begin

        arma = MA(ARMA(5.0, SA[-0.7,0.2], SA[0.7,0.2], 1.0),20)
        z = Float64[]
        for s in take(simulate(arma), 50000)
            push!(z, s)
        end
        μ, ϕ, θ, σ2 = hannan_rissanen(z,2, 2)
        @test μ ≈ 5.0 rtol = 0.05
        @test ϕ ≈ [-0.7,0.2] rtol = 0.05
        @test θ ≈ [0.7,0.2] rtol = 0.1
        @test σ2 ≈ 1.0 rtol = 0.05
        
    end

    @testset "ARIMA" begin

        arima = ARIMA{1}(3.0, SA[0.5,-0.3], SA[0.2,-0.2], 1.0)
        z = Float64[]
        for s in take(simulate(arima), 100000)
            push!(z, s)
        end
        μ, ϕ, θ, σ2 = hannan_rissanen(difference(z), 2, 2)
        @test μ ≈ 3.0 rtol = 0.05
        @test ϕ ≈ [0.5,-0.3] rtol = 0.05
        @test θ ≈ [0.2,-0.2] rtol = 0.1
        @test σ2 ≈ 1.0 rtol = 0.05
        
    end
    
end