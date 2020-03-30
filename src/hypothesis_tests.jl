import Distributions


#return a matrix of Zs of each paramenter of the model
function standard(M::Matrix,var::Matrix):
    z= M/(varˆ(1/2))
    return z
end

#Given a Z from a parameter return if he is relevant (accept Ha) or not (accept Ho)
function Walk(z::int, alfa::int)
 
    if z<=0  
        z=-z
    end if

    pvalue = ccdf(Normal, z)

    if pvalue<=alfa/2
        print("accept Ha")
    else
        print("accept Ho, B isn't relevant")
    end if 

end 


function Likelihood_ratio_test(full_model, reduced_model, Y, critical value)

    F_parameters=estimate(full_model).beta_hat
    R_parameters=estimate(reduced_model).beta_hat
    G = -2*(loglike(Y, R_parameters)-loglike(Y, F_parameters))

    if G<=0
        G=-G
    end if

    p_value = ccdf(Chisq, G)
 
    if pvalue<=alfa/2
        print("accept Ha")
    else
        print("accept Ho, B isn't relevant")
    end if

end 

print("it's working")