mutable struct Model{T <: Real} 
    y::Vector{Int}
    X::Matrix{T}
    threshold::T
    num_obs::Int
    beta_hat::Vector{T}
    pi_hat::Vector{T}
    y_hat::Vector{Int}
    dof_log::Int
    dof_resid::Int
    dof_total::Int
    llk::T
    aic::T
    bic::T
    # r2::T
    dev_residuals::Vector{T}
    dev_residuals_var::Float64
    # sigma::Matrix{T}
    # std_error::Vector{T}
    # z_value::Vector{T}
    # z_test_p_value::Vector{T}
end