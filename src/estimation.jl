using LinearAlgebra, Optim, NLSolversBase

struct Modelo{T <: Real} 
    Y::Vector{Int}
    X::VecOrMat{T}
end

function LogLike(modelo::Modelo, beta::Vector{T}) where {T<:Real}
    X = modelo.X; Y = modelo.Y
    numobs = length(Y); numpar = length(beta); X = hcat(ones(numobs), X)
    llike = sum(Y.*X*beta - log.(ones(numobs) + exp.(X*beta)))
    return llike
end

function estimate(modelo::Modelo) where {T<:Real}
    X = modelo.X; Y = modelo.Y
    numpar = length(size(X)) == 1 ? 1 : size(X)[2]
    numobs = length(Y)

    func = TwiceDifferentiable(vars -> -LogLike(modelo, vars[1:numpar+1]), ones(numpar+1); autodiff=:forward);

    opt = optimize(func, ones(numpar+1))

    # Estimando os betas e achando a variancia dos betas chapeu
    beta_chapeu = Optim.minimizer(opt)
    var_beta = inv(NLSolversBase.hessian!(func,beta_chapeu))

     # Estimando os pi's
     numobs = length(Y)
     X = hcat(ones(numobs), X)
     num = exp.(X*beta_chapeu)
     dem = ones(numobs) + exp.(X*beta_chapeu)
     πi = num./dem 

    return Dict("β" => beta_chapeu, "Σ" => var_beta, "π" => πi)
end

function AIC(modelo::Modelo)
    X = modelo.X; Y = modelo.Y
    beta_hat = estimate(modelo)["β"]; LbetaML = LogLike(modelo, beta_hat)
    p = length(beta_hat); n = length(Y)
    return -2LbetaML + 2*p 
end

function y_hat(modelo::Modelo, cut::Float64) where {T<:Real}
    X = modelo.X; Y = modelo.Y
    pi_hat = estimate(modelo)["π"]
    y_hat = map(x-> x > cut ? 1 : 0, pi_hat)
    return hcat(Y, y_hat)
end

function residuals_dev(modelo::Modelo)
    beta_hat = estimate(modelo)["β"]; pi_hat = estimate(modelo)["π"]; Y = modelo.Y; numobs = length(Y)
    return sign.(Y-pi_hat).*sqrt.(-2*Y.*log.(pi_hat) -2*(ones(numobs) - Y).*log.(ones(numobs) - pi_hat))
end

# COMPARA COM R.
y = [0,1,0,1,1,0,1,0,1,1,1,0,0,0,0,0,1,1,0,0,0,1,1,0,1,1,1,0,0,0,1,1,1,1,0,0,1,0,1,0,0,0,0,0,1,1,0,1,1,0,0,0,0,0,1,1,0,0,1,0,0,0,0,1,0,0,1,0,1,0,1,1,1,0,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,0,1,0];
x = [2.5,1.7,4,2.3,3.7,4.8,1.9,5.3,3.1,1.9,2.3,3.6,4.7,5.8,6,3.9,2.4,1.7,3.7,4.8,3.2,2.7,1.2,8.2,1.8,2.5,2.2,4,4.2,3.7,2.4,1.6,2,2.5,3.8,4.3,2,5.2,2.4,2.6,1.3,3.8,4.5,3,2.1,1.9,1.7,1.7,1.3,2.5,3.5,5.6,3.8,4,2.5,1.2,3,3,2.1,2.5,2.9,4,3.2,1.2,3.5,4,2.3,2.9,2.4,5,2.2,1.3,1.7,3,3,3.5,5.8,4.8,2.3,2.6,1.8,2.9,3.2,4.2,2.6,6,4.5,1.3,2.4,4.3,1.8,2.4];
N_dep = [3,3,2,2,4,1,3,2,4,3,4,1,2,2,4,3,4,4,2,1,2,3,3,5,1,1,3,1,1,1,2,3,1,3,1,2,2,2,3,4,2,1,0,0,2,2,4,2,3,1,2,3,2,0,1,2,1,1,2,1,1,3,3,2,3,1,3,4,2,3,3,3,3,2,2,2,2,1,3,2,2,2,1,1,1,1,3,2,2,2,0,2];
Vinculo_emp = [0,0,0,0,1,1,1,0,0,0,0,1,0,1,1,0,0,0,1,1,0,0,0,1,0,0,0,1,1,1,0,0,0,0,1,1,0,1,1,1,0,0,0,0,0,0,1,0,0,0,1,1,1,1,0,1,1,1,0,1,1,1,1,0,1,1,0,1,0,1,1,0,0,1,0,0,0,1,0,0,0,0,1,1,1,1,0,0,0,1,0,1]
X = hcat(x, N_dep);
X1 = hcat(X, Vinculo_emp);
modelo1 = Modelo(y, x); modelo2 = Modelo(y, X); modelo3 = Modelo(y, X1);

esta1 = estimate(modelo1)
LogLike(modelo1, esta1["β"])
y_hat(modelo1, .5)
AIC(modelo1)
sum(residuals_dev(modelo1).^2)

esta2 = estimate(modelo2)
LogLike(modelo2, esta2["β"])
y_hat(modelo2, .5)
AIC(modelo2)
sum(residuals_dev(modelo2).^2)

esta3 = estimate(modelo3)
LogLike(modelo3, esta3["β"])
y_hat(modelo3, .5)
AIC(modelo3)
sum(residuals_dev(modelo3).^2)

