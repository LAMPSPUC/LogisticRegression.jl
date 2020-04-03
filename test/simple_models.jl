@testset "simple models" begin
    using LogisticRegression
    y  = [0,1,0,1,1,0]
    x0 = ones(length(y))
    x1 = [2.5,1.7,4,2.3,3.7,4.8]
    x2 = [3,3,2,2,4]
    x3 = [0,0,0,0,1,1]
    X_0 = x0
    X_1 = hcat(x0, x1)
    X_2 = hcat(X_1, x2)
    X_3 = hcat(X_1, x3)

    @testset "model X_0" begin
        model_0 = logreg(y, X_0)

        @test model_0.num_obs == 6
        @test model_0.dof_log == 0
        @test model_0.dof_resid == 5
        @test model_0.dof_total == 5
        @test model_0.llk ≈ -3.365058 atol = 1e-5
        @test model_0.aic ≈ 8.7301 atol = 1e-2
        @test model_0.bic ≈ 8.339555 atol = 1e-4
        @test modelo0.pi_hat .≈ [0.5 0.5 0.5 0.5 0.5 0.5] atol = 1e-2
        @test sum(model_0.deviance_residuals.^2) ≈ 6.7301 atol = 1e-2
        @test modelo0.y_hat == [0 0 0 0 0 0 ]
    end

    @testset "model X_1" begin
        model_1 = logreg(y, X_1)

        @test model_1.num_obs == 92
        @test model_1.dof_log == 2
        @test model_1.dof_resid == 90
        @test model_1.dof_total == 91
        @test model_1.llk == 0 # It will fail
        @test model_1.aic == 0 # It will fail
        # continues
    end

    @testset "model X_2" begin

    end

    @testset "model X_3" begin

    end
end