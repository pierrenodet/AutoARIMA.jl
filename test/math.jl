using Test, AutoARIMA

@testset "Mathematical Properties" begin

    @test reduce(&, yule_walker(seriesF, 15) .≈ levinson_durbin(seriesF, 15))
    
    @testset "MA(q) forecast after q steps should be equal to the constant" begin
        μ = 0.1
        ma = MAModel{3}(μ,[0.2,0.3,-0.5],1.0)
        z = [0.2,0.5,0.1]
        f = Float64[]
        for i in 1:10
            zhat = forecast(ma,z)
            push!(z,zhat)
            push!(f,zhat)
        end
        @test all(f[4:end] .≈ μ)
    end

end