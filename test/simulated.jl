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

    @testset "ARwithHR" begin

        ar = AR(0.0, SA[0.8,-0.2], 1.0)
        z = Float64[]
        for s in take(simulate(ar), 10000)
            push!(z, s)
        end
        μ, ϕ, θ, σ2 = hannan_rissanen(z, 2, 0)
        @test μ ≈ 0.0 atol = 0.05
        @test ϕ ≈ [0.8,-0.2] rtol = 0.05
        @test σ2 ≈ 1.0 rtol = 0.05

    end

    @testset "MAwithHR" begin

        ma = MA(3.0, SA[0.2,-0.2], 1.0)
        z = Float64[]
        for s in take(simulate(ma), 10000)
            push!(z, s)
        end
        μ, ϕ, θ, σ2 = hannan_rissanen(z, 0, 2)
        @test μ ≈ 3.0 rtol = 0.05
        @test θ ≈ [0.2,-0.2] rtol = 0.05
        @test σ2 ≈ 1.0 rtol = 0.05
        
    end

    # @testset "ARMA" begin

    #     arma = ARMA(3.0, SA[0.5,-0.3],SA[0.2,-0.2], 1.0)
    #     z = Float64[]
    #     for s in take(simulate(arma), 10000)
    #         push!(z, s)
    #     end
    #     μ, ϕ, θ, σ2 = hannan_rissanen(z, 2, 2)
    #     @test μ ≈ 3.0 rtol = 0.05
    #     @test ϕ ≈ [0.5,-0.3] rtol = 0.05
    #     @test θ ≈ [0.2,-0.2] rtol = 0.05
    #     @test σ2 ≈ 1.0 rtol = 0.05
        
    # end
    
end