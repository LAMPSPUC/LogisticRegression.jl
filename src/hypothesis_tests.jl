import Distributions.ccdf
using LogisticRegression

#return a matrix of Zs of each paramenter of the model
function standard(M::Matrix,var::Matrix)
    z= M/(varˆ(1/2))
    return z
end

println("it's working")

#Given a Z from a parameter return if he is relevant (accept Ha) or not (accept Ho)
function Walk(z::Int, alfa::Int)
 
    if z<=0  
        z=-z
    end

    pvalue = ccdf(Normal, z)

    if pvalue<=alfa/2
        print("accept Ha")
    else
        print("accept Ho, B isn't relevant")
    end 

end 

println("it's working")


function Likelihood_ratio_test(full_model::Modelo{T}, reduced_model::Modelo{T},Y::Vectot{Int}, alfa::Number)

    F_parameters=estimate(full_model).beta_hat
    R_parameters=estimate(reduced_model).beta_hat
    G = -2*(loglike(Y, R_parameters)-loglike(Y, F_parameters))

    if G<=0
        G=-G
    end

    p_value = ccdf(Chisq, G)
 
    if pvalue<=alfa/2
        print("accept Ha")
    else
        print("accept Ho, B isn't relevant")
    end

end 


println("it's working")