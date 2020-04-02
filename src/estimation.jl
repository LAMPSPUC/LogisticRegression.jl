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
    return -2/length(num_obs) * llk + 2 * dof_log / length(num_obs) 

function eval_bic(llk::T, dof_log::Int, num_obs::Int) where T
    return length(num_obs)) * llk + dof_log * log(length(num_obs)))
end

function eval_r2(ll1::T, ll0::T) where T
    return 1-ll1/ll0
end


function logreg(y::Vector{Number}, X::Vector{T}) where T
    return logreg(y, X[:, :])
end

function sigma(y::Vector{Int}, X::Matrix{T}, beta_hat::Vector{T}, num_obs::Int) where T
    
    numpar = length(size(X)) == 1 ? 1 : size(X)[2]
    func = TwiceDifferentiable(vars -> -loglik(y, X, beta_hat, num_obs), ones(numpar+1); autodiff=:forward)
    return inv(NLSolversBase.hessian!(func,beta_hat))
    
end

function deviance_residuals(y::Vector{Int}, prevision::Vector{T}) where T
    return sign(y.-prevision).*(-2.*(y.*log(prevision)+(1.-y).*log(1.-prevision)))^(1/2)
    

function deviance_residuals_variance(dev::Vector{T}) where T
    for resi in dev
        x = (resi - mean(dev))^(2)
    end
    return x/(lenght(dev)-1)
end

function std_error(sigma::Matrix)
    return sigma.^(1/2)
end

"""
    logreg(y::Vector{Int}, X::Matrix{T}) where T 

Performs logistic regression.
"""
function logreg(y::Vector{Int}, X::Matrix{T}) where T 

    # Faz todas as funções necessárias

    return Model(y, X, ...)
end