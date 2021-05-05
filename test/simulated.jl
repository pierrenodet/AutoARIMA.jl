using Test, AutoARIMA, Base.Iterators

@testset "Simulated" begin

    @testset "AR" begin

        ar = ARModel{2}(0.0, [0.8,-0.2], 0.2)
        z = Float64[]
        for s in take(simulate(ar), 50000)
            push!(z, s)
        end
        ar = fit(ARParams(2),z)
        @test ar.μ ≈ 0.0 atol = 0.1
        @test ar.ϕ ≈ [0.8,-0.2] rtol = 0.1
        @test ar.σ2 ≈ 0.2 rtol = 0.1

    end

    @testset "AR with HR" begin

        ar = ARModel{2}(0.0, [0.8,-0.2], 0.2)
        z = Float64[]
        for s in take(simulate(ar), 50000)
            push!(z, s)
        end
        arma = fit(ARMAParams(2,0),z)
        @test arma.μ ≈ 0.0 atol = 0.1
        @test arma.ϕ ≈ [0.8,-0.2] rtol = 0.1
        @test arma.σ2 ≈ 0.2 rtol = 0.05

    end

    @testset "MA" begin

        ma = MAModel{2}(1.0, [0.2,-0.2], 0.2)
        z = Float64[]
        for s in take(simulate(ma), 50000)
            push!(z, s)
        end
        ma = fit(MAParams(2),z,n=5)
        @test ma.θ ≈ [0.2,-0.2] atol = 0.15
        @test ma.σ2 ≈ 0.2 rtol = 0.1
        
    end

    @testset "MA with HR" begin

        ma = MAModel{2}(1.0, [0.2,-0.2], 0.2)
        z = Float64[]
        for s in take(simulate(ma), 50000)
            push!(z, s)
        end
        arma = fit(ARMAParams(0,2), z)
        @test arma.μ ≈ 1.0 rtol = 0.1
        @test arma.θ ≈ [0.2,-0.2] rtol = 0.1
        @test arma.σ2 ≈ 0.2 rtol = 0.1
        
    end

    @testset "ARMA" begin

        arma = ARMAModel{2,2}(1.0, [-0.2,0.2], [0.2,0.2], 0.2)
        z = Float64[]
        for s in take(simulate(arma), 50000)
            push!(z, s)
        end
        arma= fit(ARMAParams(2,2), z)
        @test arma.μ ≈ 1.0 rtol = 0.1
        @test arma.ϕ ≈ [-0.2,0.2] rtol = 0.2
        @test arma.θ ≈ [0.2,0.2] rtol = 0.2
        @test arma.σ2 ≈ 0.2 rtol = 0.1
        
    end

    @testset "ARMA with AR Inf" begin

        ar = AR∞(ARMAModel{2,2}(1.0,[-0.2,0.2], [0.2,0.2], 0.2), 20)
        z = Float64[]
        for s in take(simulate(ar), 50000)
            push!(z, s)
        end
        arma = fit(ARMAParams(2,2), z, n=2)
        @test arma.μ ≈ 1.0 rtol = 0.1
        @test arma.ϕ ≈ [-0.2,0.2] rtol = 0.2
        @test arma.θ ≈ [0.2,0.2] rtol = 0.2
        @test arma.σ2 ≈ 0.2 rtol = 0.1
        
    end

    @testset "ARMA with MA Inf" begin

        ma = MA∞(ARMAModel{2,2}(1.0,[-0.2,0.2], [0.2,0.2], 0.2), 20)
        z = Float64[]
        for s in take(simulate(ma), 50000)
            push!(z, s)
        end
        arma = fit(ARMAParams(2,2), z, n=2)
        @test arma.μ ≈ 1.0 rtol = 0.1
        @test arma.ϕ ≈ [-0.2,0.2] rtol = 0.2
        @test arma.θ ≈ [0.2,0.2] rtol = 0.2
        @test arma.σ2 ≈ 0.2 rtol = 0.1
        
    end

    @testset "ARIMA" begin

        arima = toarma(ARIMAParams(2, 1, 2), ARMAModel{2,2}(2.0, [0.5,-0.3], [0.2,-0.2], 0.2))
        z = Float64[]
        for s in take(simulate(arima), 50000)
            push!(z, s)
        end
        arma = fit(ARMAParams(2,2),difference(z,d=1),n=2)
        @test arma.ϕ ≈ [0.5,-0.3] rtol = 0.2
        @test arma.θ ≈ [0.2,-0.2] rtol = 0.2
        @test arma.σ2 ≈ 0.2 atol = 0.1
        
    end
    
end