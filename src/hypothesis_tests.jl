function z_value(M::Vector{T}, std::Vector{T}) where T
    z= M./std
    return z
end

function calc_p_value(z::Vector{T}, num_par::Int) where {T<:Real}
    pvalue = zeros(num_par)
    for (index, value) in enumerate(z)
        if value<=0  
            value=-value
        end
        pvalue[index] = value
    end
    return 2*ccdf.(Normal(), pvalue)           
end