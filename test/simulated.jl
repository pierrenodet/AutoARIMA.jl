using Test, AutoARIMA, StaticArrays, Base.Iterators

@testset "Simulated" begin

    @testset "AR" begin

        ar = AR(0.0, SA[0.8,-0.2], 0.2)
        z = Float64[]
        for s in take(simulate(ar), 2000)
            push!(z, s)
        end
        μ, ϕ, σ2 = levinson_durbin(z, 2)
        @test μ ≈ 0.0 atol = 0.1
        @test ϕ ≈ [0.8,-0.2] rtol = 0.1
        @test σ2 ≈ 0.2 rtol = 0.1

    end

    @testset "AR with HR" begin

        ar = AR(0.0, SA[0.8,-0.2], 0.2)
        z = Float64[]
        for s in take(simulate(ar), 1000)
            push!(z, s)
        end
        μ, ϕ, θ, σ2 = hannan_rissanen(z, 2, 0, n=10)
        @test μ ≈ 0.0 atol = 0.1
        @test ϕ ≈ [0.8,-0.2] rtol = 0.1
        @test σ2 ≈ 0.2 rtol = 0.1

    end

    @testset "MA with HR" begin

        ma = MA(1.0, SA[0.2,-0.2], 0.2)
        z = Float64[]
        for s in take(simulate(ma), 10000)
            push!(z, s)
        end
        μ, ϕ, θ, σ2 = hannan_rissanen(z, 0, 2, n=10)
        @test μ ≈ 1.0 rtol = 0.1
        @test θ ≈ [0.2,-0.2] rtol = 0.1
        @test σ2 ≈ 0.2 rtol = 0.1
        
    end

    @testset "ARMA" begin

        arma = ARMA(5.0, SA[-0.7,0.2], SA[0.7,0.2], 0.2)
        z = Float64[]
        for s in take(simulate(arma), 50000)
            push!(z, s)
        end
        μ, ϕ, θ, σ2 = hannan_rissanen(z, 2, 2, n=10)
        @test μ ≈ 5.0 rtol = 0.1
        @test ϕ ≈ [-0.7,0.2] rtol = 0.1
        @test θ ≈ [0.7,0.2] rtol = 0.1
        @test σ2 ≈ 0.2 rtol = 0.1
        
    end

    @testset "ARMA with MA Inf" begin

        arma = MA(ARMA(5.0, SA[-0.7,0.2], SA[0.7,0.2], 0.2), 20)
        z = Float64[]
        for s in take(simulate(arma), 50000)
            push!(z, s)
        end
        μ, ϕ, θ, σ2 = hannan_rissanen(z, 2, 2, n=10)
        @test μ ≈ 5.0 rtol = 0.1
        @test ϕ ≈ [-0.7,0.2] rtol = 0.1
        @test θ ≈ [0.7,0.2] rtol = 0.1
        @test σ2 ≈ 0.2 rtol = 0.1
        
    end

    @testset "ARIMA" begin

        arima = ARIMA{1}(2.0, SA[0.5,-0.3], SA[0.2,-0.2], 0.2)
        z = Float64[]
        for s in take(simulate(arima), 50000)
            push!(z, s)
        end
        μ, ϕ, θ, σ2 = hannan_rissanen(difference(z), 2, 2, n=10)
        @test μ ≈ 2.0 atol = 0.1
        @test ϕ ≈ [0.5,-0.3] rtol = 0.1
        @test θ ≈ [0.2,-0.2] rtol = 0.1
        @test σ2 ≈ 0.2 atol = 0.1
        
    end
    
end