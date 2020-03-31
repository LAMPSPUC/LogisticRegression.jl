
import Distributions.ccdf
using LogisticRegression

#return a matrix of Zs of each paramenter of the model
function standard(M::Vector, vari_::Vector)
    z= M./(vari_.^(1/2))
    return z
end

println("it's working")

#Given a Z from a parameter return if he is relevant (accept Ha) or not (accept Ho)
function Walk(z::Number, alfa::Number)
 
    if z<=0  
        z=-z
    end

    pvalue = ccdf(Normal(), z)
    print(pvalue)

    if pvalue <= alfa/2
        print("accept Ha, B is relevant")
        return 0
    else
        print("accept H0, B isn't relevant")
        return 1
    end 

end 

println("it's working")



function Likelihood_ratio_test(full_model::LogisticRegression.Modelo{T}, reduced_model::LogisticRegression.Modelo{T}, alfa::Number) where {T<:Real}

    F_parameters=estimate(full_model).beta_hat

    R_parameters=estimate(reduced_model).beta_hat
    G = -2*(loglike(Modelo.Y, R_parameters)-loglike(Y, F_parameters))

    p_value = ccdf(Chisq, G)
 
    if pvalue<=alfa/2
        print("accept Ha")
        return 0
    else
        print("accept Ho, B isn't relevant")
        return 1
    end

end 


println("it's working")