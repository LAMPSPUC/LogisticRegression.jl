@testset "simple models" begin
    using LogisticRegression
    y  = [0,1,0,1,1,0]
    x0 = ones(length(y))
    x1 = [2.5,1.7,4,2.3,3.7,4.8]
    x2 = [3,3,2,2,4,1]
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
        @test model_0.beta_hat ≈ [0.0000] atol = 1e-5
        @test model_0.llk ≈ -4.158883 atol = 1e-5
        @test model_0.dev_residuals ≈ [-1.17741, 1.17741, -1.17741, 1.17741,  1.17741, -1.17741] atol = 1e-5
        @test model_0.aic ≈ 10.318 atol = 1e-3
        @test model_0.bic ≈ 10.10953 atol = 1e-4
        @test model_0.pi_hat ≈ [0.5, 0.5, 0.5, 0.5, 0.5, 0.5] atol = 1e-2
        @test model_0.y_hat == ones(Int64, 6)
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
        modelo_2 = logreg(y, X_2)

        @test modelo_2.num_obs == 6
        @test modelo_2.dof_log == 2
        @test modelo_2.dof_resid == 3
        @test modelo_2.dof_total == 5
        @test modelo_2.beta_hat ≈ [0.7187, -1.2929, 1.2082] atol = 1e-4
        @test modelo_2.llk ≈ -2.639604 atol = 1e-5
        @test modelo_2.dev_residuals ≈ [-1.6707271, 0.4704612, -0.4952902, 1.1096542, 0.8731406, -0.1659078] atol = 1e-7
        @test modelo_2.aic ≈ 11.279 atol = 1e-3
        @test modelo_2.bic ≈ 10.65449 atol = 1e-4
        @test modelo_2.pi_hat ≈ [0.75233160, 0.89523692, 0.11543226, 0.54028085, 0.68304996, 0.01366843] atol = 1e-7
        @test modelo_2.y_hat == [1, 1, 0, 1, 1, 0]
    end

    @testset "model X_3" begin

    end
end