# This function should find beta_hat
function maxllk(y::Vector{Int}, X::VecOrMat{T}, numpar::Int) where {T<:Real}
    func = TwiceDifferentiable(vars -> -loglik(y, X, vars[1:numpar], length(y)), ones(numpar); autodiff=:forward);
    return optimize(func, ones(numpar))
end

function fit(opt)
    return Optim.minimizer(opt)
end

function eval_pi_hat(y::Vector{Int}, X::VecOrMat{T}, beta_hat::Vector{T}, numobs) where {T<:Real}
    return exp.(X*beta_hat)./( ones(numobs) + exp.(X*beta_hat) )
end

function eval_y_hat(pi_hat::Vector{Float64}, threshold::Float64)
    map(x-> x > threshold ? 1 : 0,  pi_hat)
end

function loglik(y::Vector{Int}, X::VecOrMat{T}, beta_hat::Vector{S}, num_obs::Int) where {T<:Real, S} 
    return sum(y .* X * beta_hat - log.(ones(num_obs) + exp.(X * beta_hat)))
end

function eval_aic(llk::T, dof_log::Int) where {T<:Real}
    return 2*log(llk) + 2*dof_log
end

function eval_bic(llk::T, dof_log::Int, y::Vector{Int}) where {T<:Real}
    return 2*log(llk) + dof_log*log(y)
end


function logreg(y::Vector{Int}, X::Vector{T}) where {T<:Real}
    return logreg(y, X[:, :])
end

"""
    logreg(y::Vector{Int}, X::Matrix{T}) where T 

Performs logistic regression.
"""
function logreg(y::Vector{Int}, X::Matrix{T}, threshold::Float64 = .5) where T 
    num_obs, num_par = size(X) # esse num_par nao esta contando o beta0

    # Faz todas as funções necessárias
    maxllk    = maxllk(y, X, num_par)
    beta_hat  = fit(maxllk)
    dof_log   = num_par + 1 # +1 é par contar o beta0
    dof_resid = num_obs - num_par
    dof_total = num_obs - 1
    pi_hat    = eval_pi_hat(y, X, beta_hat, num_obs)
    y_hat     = eval_y_hat(pi_hat, threshold)
    llk       = loglik(y, X, beta_hat)
    aic       = eval_aic(llk, dof_log)
    bic       = eval_bic(llk::T, dof_log::Int, y)
    return Model(y, X, ...)
end