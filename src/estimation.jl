# This function should find beta_hat
function fit(y, X, ...)

end

function eval_dof_log()
    
end

function eval_dof_resid()
    
end

function eval_dof_total()
    
end

function eval_pi_hat()
    
end

function eval_y_hat()
    
end

function loglik(y::Vector{Int}, X::Matrix{T}, beta_hat::Vector{T}, num_obs::Int) where T
    return sum(y .* X * beta_hat - log.(ones(num_obs) + exp.(X * beta_hat)))
end

function eval_aic(llk::T, dof_log::Int, num_obs::Int) where T
    return 2 * dof_log - 2 * llk 

function eval_bic(llk::T, dof_log::Int, num_obs::Int) where T
    return dof_log * log(num_obs) - 2 * llk 
end

function eval_r2(ll1::T, ll0::T) where T
    return 1 - ll1/ll0
end

function sigma(y::Vector{Int}, X::Matrix{T}, beta_hat::Vector{T}, num_obs::Int, num_par::Int) where T
    func = TwiceDifferentiable(vars -> -loglik(y, X, beta_hat, num_obs), ones(num_par); autodiff=:forward)
    return inv(NLSolversBase.hessian!(func, beta_hat))
end

function deviance_residuals(y::Vector{Int}, prevision::Vector{T}) where T
    return sign(y.-prevision).*(-2 .* (y.*log(prevision)+(1 .- y).*log(1 .- prevision)))^(1/2)
end

function deviance_residuals_variance(dev::Vector{T}) where T
    mu = mean(dev)
    variance = zero(T)
    for resi in dev
        variance += (resi - mu)^(2)
    end
    return variance/(length(dev)-1)
end

function std_error(sigma::Matrix{T}, num_par::Int) where T
    std_err_vec = Vector{T}(undef, num_par)
    for i in 1:num_par
        std_err_vec[i] = sqrt(sigma[i, i])
    end
    return std_err_vec
end

function logreg(y::Vector{T}, X::Vector{T}) where T
    return logreg(y, X[:, :])
end

"""
    logreg(y::Vector{Int}, X::Matrix{T}) where T 

Performs logistic regression.
"""
function logreg(y::Vector{Int}, X::Matrix{T}) where T 

    # Faz todas as funções necessárias

    return Model(y, X, ...)
end