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

function eval_aic(llk::T, dof_log::Int) where T
    
end

function eval_bic(llk::T, dof_log::Int) where T
    
end


function logreg(y::Vector{Int}, X::Vector{T}) where T
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