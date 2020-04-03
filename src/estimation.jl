# This function should find beta_hat
function maxllk(y::Vector{Int}, X::Matrix{T}, num_par::Int, num_obs::Int) where T <: Real
    func = TwiceDifferentiable(vars -> -optim_loglik(y, X, vars, num_obs), ones(num_par); autodiff=:forward)
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

function optim_loglik(y::Vector{Int}, X::Matrix{T}, beta_hat::Vector{T}, num_obs::Int) where T <: Real 
    return sum(y .* X * beta_hat - log.(ones(num_obs) + exp.(X * beta_hat)))
end

function eval_loglik(opt)
    return Optim.minimum(opt)
end

function logreg(y::Vector{Int}, X::Vector{T}) where {T<:Real}
    return logreg(y, X[:, :])
end

"""
    logreg(y::Vector{Int}, X::Matrix{T}) where T 

Performs logistic regression.
"""
function logreg(y::Vector{Int}, X::Matrix{T}, threshold::Float64 = 0.5) where T 
    num_obs, num_par = size(X)
    # Faz todas as funções necessárias
    maxllk    = maxllk(y, X, num_par)
    beta_hat  = eval_beta_hat(maxllk)
    dof_log   = num_par
    dof_resid = num_obs - num_par
    dof_total = num_obs - 1
    pi_hat    = eval_pi_hat(y, X, beta_hat, num_obs)
    y_hat     = eval_y_hat(pi_hat, threshold)
    llk       = loglik(y, X, beta_hat)
    aic       = eval_aic(llk, dof_log)
    bic       = eval_bic(llk, dof_log, y)
    return Model(y, X, ...)
end