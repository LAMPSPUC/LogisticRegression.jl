# This function should find beta_hat
function maxllk(y::Vector{Int}, X::Matrix{T}, num_par::Int, num_obs::Int) where T <: Real
    func = Optim.TwiceDifferentiable(vars -> -optim_loglik(y, X, vars, num_obs), ones(num_par); autodiff=:forward)
    return optimize(func, ones(num_par))
end

function eval_beta_hat(opt)
    return Optim.minimizer(opt)
end

function eval_pi_hat(y::Vector{Int}, X::Matrix{T}, beta_hat::Vector{T}, num_obs::Int) where T <: Real
    return exp.(X*beta_hat)./(ones(num_obs) + exp.(X*beta_hat))
end

function eval_y_hat(pi_hat::Vector{T}, threshold::T) where T <: Real
    return map(x -> x > threshold ? 1 : 0, pi_hat)
end

function optim_loglik(y::Vector{Int}, X::Matrix{T}, beta_hat::Vector{S}, num_obs::Int) where {T <: Real, S}
    return sum(y .* X * beta_hat - log.(ones(num_obs) + exp.(X * beta_hat)))
end

function eval_loglik(opt)
    return -Optim.minimum(opt)
end

function eval_aic(llk::T, dof_log::Int) where T
    return 2 * (dof_log + 1) - 2 * llk 
end

function eval_bic(llk::T, dof_log::Int, num_obs::Int) where T
    return (dof_log + 1) * log(num_obs) - 2 * llk 
end

function eval_r2(ll1::T, ll0::T) where T
    return 1 - ll1/ll0
end

function eval_sigma(y::Vector{Int}, X::Matrix{T}, beta_hat::Vector{T}, num_obs::Int, num_par::Int)  where {T<:Real}
    func = TwiceDifferentiable(vars -> -optim_loglik(y, X, vars, num_obs), ones(num_par); autodiff=:forward)
    return inv(NLSolversBase.hessian!(func, beta_hat))
end

function eval_std_error(sigma::Matrix{T}, num_par::Int) where {T<:Real}
    std_error = zeros(num_par)
    for i=1:num_par
        std_error[i,1]=sigma[i, i]^(.5)
    end
    return std_error
end

function deviance_residuals(y::Vector{Int}, pi_hat::Vector{T}) where T
    return sign.(y.-pi_hat) .* (-2 .* (y .* log.(pi_hat) + (1 .- y) .* log.(1 .- pi_hat))).^(1/2)
end

function deviance_residuals_variance(dev::Vector{T}) where T
    mu = mean(dev)
    variance = zero(T)
    for resi in dev
        variance += (resi - mu)^2
    end
    return variance/(length(dev)-1)
end

function logreg(y::Vector{Int}, X::Vector{T}; threshold::Float64 = 0.5) where T
    return logreg(y, X[:, :]; threshold = threshold)
end

"""
    logreg(y::Vector{Int}, X::Matrix{T}) where T 

Performs logistic regression.
"""
function logreg(y::Vector{Int}, X::Matrix{T}; threshold::T = 0.5) where T 
    num_obs, num_par = size(X)
    # Faz todas as funções necessárias
    opt               = maxllk(y, X, num_par, num_obs)
    beta_hat          = eval_beta_hat(opt)
    dof_log           = num_par - 1
    dof_resid         = num_obs - num_par
    dof_total         = num_obs - 1
    pi_hat            = eval_pi_hat(y, X, beta_hat, num_obs)
    y_hat             = eval_y_hat(pi_hat, threshold)
    llk               = eval_loglik(opt)
    aic               = eval_aic(llk, dof_log)
    bic               = eval_bic(llk, dof_log, num_obs)
    dev_residuals     = deviance_residuals(y, pi_hat)
    dev_residuals_var = deviance_residuals_variance(dev_residuals)
    sigma             = eval_sigma(y, X, beta_hat, num_obs, num_par)
    std_error         = eval_std_error(sigma, num_par)
    # zvalue            = z_value(beta_hat, std_error)
    # z_test_p_value    = calc_p_value(zvalue, num_par)

    return Model(y, X, threshold, num_obs, beta_hat, pi_hat, y_hat, dof_log, dof_resid,
                dof_total, llk, aic, bic, dev_residuals, dev_residuals_var, sigma, std_error)
end