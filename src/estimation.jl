using LinearAlgebra, Optim, NLSolversBase

function ensure_is_matrix(y::Vector{T}) where T
    return y[:, :]
end
function ensure_is_matrix(y::Matrix{T}) where T
    return y
end

struct Modelo{T} 
    Y::Vector{Int}
    X::Matrix{T}

    function Modelo(Y::Vector{Int}, X::VecOrMat{T}) where {T<:Real}
        X = ensure_is_matrix(X)
        return new{T}(Y, X) 
    end
end

struct Estimations{T}
    beta_hat::Vector{T}
    sigma::Matrix{T}
    pi_hat::Vector{Float64}
end

function loglike(modelo::Modelo, beta::Vector{T2}) where {T1<:Real, T2<:Real}
    X = modelo.X
    Y = modelo.Y
    numobs = length(Y); numpar = length(beta); X = hcat(ones(numobs), X)
    llike = sum(Y.*X*beta - log.(ones(numobs) + exp.(X*beta)))
    return llike
end

function estimate(modelo::Modelo{T}) where {T<:Real}
    X = modelo.X
    Y = modelo.Y
    numpar = length(size(X)) == 1 ? 1 : size(X)[2]
    numobs = length(Y)

    func = TwiceDifferentiable(vars -> -loglike(modelo, vars[1:numpar+1]), ones(numpar+1); autodiff=:forward);

    opt = optimize(func, ones(numpar+1))

    # Estimando os betas e achando a variancia dos betas chapeu
    beta_hat = Optim.minimizer(opt)
    var_beta = inv(NLSolversBase.hessian!(func,beta_hat))

     # Estimando os pi's
     X = hcat(ones(numobs), X)
     num = exp.(X*beta_hat)
     dem = ones(numobs) + exp.(X*beta_hat)
     πi = num./dem 

    return Estimations(beta_hat, var_beta, πi)
end

function aic(modelo::Modelo{T}) where {T<:Real}
    Y = modelo.Y
    beta_hat = estimate(modelo).beta_hat; LbetaML = loglike(modelo, beta_hat)
    p = length(beta_hat); n = length(Y)
    return -2LbetaML + 2*p 
end

function y_hat(modelo::Modelo{T}, threshold::Float64) where {T<:Real}
    Y = modelo.Y
    pi_hat = estimate(modelo).pi_hat
    y_hat = map(x-> x > threshold ? 1 : 0, pi_hat)
    return hcat(Y, y_hat)
end

function residuals_dev(modelo::Modelo{T}) where {T<:Real}
    estimations = estimate(modelo)
    beta_hat = estimations.beta_hat
    pihat = estimations.pi_hat
    Y = modelo.Y
    numobs = length(Y)
    return sign.(Y-pihat).*sqrt.(-2*Y.*log.(pihat) -2*(ones(numobs) - Y).*log.(ones(numobs) - pihat))
end
