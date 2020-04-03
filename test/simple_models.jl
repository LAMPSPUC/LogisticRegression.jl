@testset "simple models" begin
    using LogisticRegression
    y  = [0,1,0,1,1,0,1,0,1,1,1,0,0,0,0,0,1,1,0,0,0,1,1,0,1,1,1,0,0,0,1,1,1,1,0,0,1,0,1,0,0,0,0,0,1,1,0,1,1,0,0,0,0,0,1,1,0,0,1,0,0,0,0,1,0,0,1,0,1,0,1,1,1,0,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,0,1,0]
    x0 = ones(length(y))
    x1 = [2.5,1.7,4,2.3,3.7,4.8,1.9,5.3,3.1,1.9,2.3,3.6,4.7,5.8,6,3.9,2.4,1.7,3.7,4.8,3.2,2.7,1.2,8.2,1.8,2.5,2.2,4,4.2,3.7,2.4,1.6,2,2.5,3.8,4.3,2,5.2,2.4,2.6,1.3,3.8,4.5,3,2.1,1.9,1.7,1.7,1.3,2.5,3.5,5.6,3.8,4,2.5,1.2,3,3,2.1,2.5,2.9,4,3.2,1.2,3.5,4,2.3,2.9,2.4,5,2.2,1.3,1.7,3,3,3.5,5.8,4.8,2.3,2.6,1.8,2.9,3.2,4.2,2.6,6,4.5,1.3,2.4,4.3,1.8,2.4]
    x2 = [3,3,2,2,4,1,3,2,4,3,4,1,2,2,4,3,4,4,2,1,2,3,3,5,1,1,3,1,1,1,2,3,1,3,1,2,2,2,3,4,2,1,0,0,2,2,4,2,3,1,2,3,2,0,1,2,1,1,2,1,1,3,3,2,3,1,3,4,2,3,3,3,3,2,2,2,2,1,3,2,2,2,1,1,1,1,3,2,2,2,0,2]
    x3 = [0,0,0,0,1,1,1,0,0,0,0,1,0,1,1,0,0,0,1,1,0,0,0,1,0,0,0,1,1,1,0,0,0,0,1,1,0,1,1,1,0,0,0,0,0,0,1,0,0,0,1,1,1,1,0,1,1,1,0,1,1,1,1,0,1,1,0,1,0,1,1,0,0,1,0,0,0,1,0,0,0,0,1,1,1,1,0,0,0,1,0,1]
    X_0 = x0
    X_1 = hcat(x0, x1)
    X_2 = hcat(X_1, x2)
    X_3 = hcat(X_1, x3)

    @testset "model X_0" begin
        model_0 = logreg(y, X_0)
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