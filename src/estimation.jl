# This function should find beta_hat
function fit(y, X, numpar)
    func = TwiceDifferentiable(vars -> -loglik(y, X, vars[1:numpar], length(y)), ones(numpar); autodiff=:forward);
    opt  = optimize(func, ones(numpar))
    return Optim.minimizer(opt)
end

function eval_dof_log(beta_hat)
    return length(beta_hat)
end

function eval_dof_resid(y, beta_hat)
    return length(y) - length(beta_hat)
end

function eval_dof_total(y, beta_hat)
    return length(y) - 1
end

function eval_pi_hat(y, X, beta_hat, numobs)
    exp.(X*beta_hat)./( ones(numobs) + exp.(X*beta_hat) )
end

function eval_y_hat(pi_hat, threshold)
    map(x-> x > threshold ? 1 : 0,  pi_hat)
end

function loglik(y::Vector{Int}, X::VecOrMat{T}, beta_hat::Vector{S}, num_obs::Int) where {T<:Real, S} 
    return sum(y .* hcat(ones(num_obs), X) * beta_hat - log.(ones(num_obs) + exp.(hcat(ones(num_obs), X) * beta_hat)))
end

function eval_aic(llk::T, dof_log::Int) where T
    return 2*log(llk) + 2*dof_log
end

function eval_bic(llk::T, dof_log::Int, y::Vector{Int}) where T
    return 2*log(llk) + dof_log*log(y)
end


function logreg(y::Vector{Int}, X::Vector{T}) where T
    return logreg(y, X[:, :])
end

"""
    logreg(y::Vector{Int}, X::Matrix{T}) where T 

Performs logistic regression.
"""
function logreg(y::Vector{Int}, X::Matrix{T}, threshold::Float64 = .5) where T 
    numobs, numpar = size(X)

    # Faz todas as funções necessárias
    beta_hat  = fit(y, X)
    num_obs   = length(y)
    dof_log   = eval_dof_log(y)
    dof_resid = eval_dof_resid(y, beta_hat)
    dof_total = eval_dof_total(y, beta_hat)
    pi_hat    = eval_pi_hat(y, X, beta_hat, numobs)
    y_hat     = eval_y_hat(pi_hat, threshold)
    llk       = loglik(y, X, beta_hat)
    aic       = eval_aic(llk, dof_log)
    bic       = eval_bic(llk::T, dof_log::Int, y)
    return Model(y, X, ...)
end