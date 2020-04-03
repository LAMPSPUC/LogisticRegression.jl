@testset "simple models" begin
    y  = [0,1,0,1,1,0]
    x0 = ones(length(y))
    x1 = [2.5,1.7,4,2.3,3.7,4.8]
    x2 = [3,3,2,2,4,1]
    x3 = [0,0,0,0,1,1]
    X_0 = x0
    X_1 = hcat(x0, x1)
    X_2 = hcat(X_1, x2)
    X_3 = hcat(X_2, x3)

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
        @test model_0.std_error ≈ 0.8165 atol = 1e-3
    end

    @testset "model X_1" begin
        model_1 = logreg(y, X_1) 
        @test model_1.num_obs == 6
        @test model_1.beta_hat ≈ [3.971,-1.260]  atol = 1e-3
        @test model_1.dof_log == 1
        @test model_1.dof_resid == 4
        @test model_1.dof_total == 5
        @test model_1.pi_hat ≈ [0.6940996, 0.8614932, 0.2551409, 0.7448733, 0.3333144, 0.1110785] atol = 1e-3
        @test model_1.y_hat ≈ [1, 1 , 0, 1, 0, 0] atol = 1e-3
        @test model_1.llk ≈ -3.1391 atol = 1e-3 
        @test model_1.aic ≈  10.2782 atol = 1e-3 
        @test model_1.bic ≈ 9.86172 atol = 1e-3 
        @test model_1.dev_residuals ≈  [-1.5391528,  0.5460551, -0.7675418,  0.7675169,  1.4823421, -0.4852760] atol = 1e-3
        @test model_1.std_error ≈ [3.353, 1.026] atol = 1e-3
    end

    @testset "model X_2" begin
        model_2 = logreg(y, X_2)

        @test model_2.num_obs == 6
        @test model_2.dof_log == 2
        @test model_2.dof_resid == 3
        @test model_2.dof_total == 5
        @test model_2.beta_hat ≈ [0.7187, -1.2929, 1.2082] atol = 1e-4
        @test model_2.llk ≈ -2.639604 atol = 1e-5
        @test model_2.dev_residuals ≈ [-1.6707271, 0.4704612, -0.4952902, 1.1096542, 0.8731406, -0.1659078] atol = 1e-7
        @test model_2.aic ≈ 11.279 atol = 1e-3
        @test model_2.bic ≈ 10.65449 atol = 1e-4
        @test model_2.pi_hat ≈ [0.75233160, 0.89523692, 0.11543226, 0.54028085, 0.68304996, 0.01366843] atol = 1e-7
        @test model_2.y_hat == [1, 1, 0, 1, 1, 0]
        @test model_2.std_error ≈ [4.8242, 1.3811, 1.3498]atol = 1e-3
    end

    @testset "model X_3" begin
        model_3 = logreg(y, X_3) 
        @test model_3.num_obs == 6
        @test model_3.dof_log == 3
        @test model_3.dof_resid == 2
        @test model_3.dof_total == 5
        @test model_3.pi_hat ≈ [8.114495e-11, 1.000000e+00, 2.220446e-16, 1.000000e+00, 1.000000e+00, 4.127692e-11] atol = 1e-3
        @test model_3.y_hat ≈ [0, 1, 0, 1, 1, 0] atol = 1e-3
        @test model_3.llk ≈ -2.482672e-10 atol = 1e-3 
        @test model_3.aic ≈ 8 atol = 1e-3 
        @test model_3.bic ≈ 7.167038 atol = 1e-3 
    end
end