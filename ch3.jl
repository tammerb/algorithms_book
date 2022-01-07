#= Algorithm 3.1. An implementation
of the factor product, which constructs the factor representing the
joint distribution of two smaller factors ϕ and ψ. If we want to compute
the factor product of ϕ and ψ, we
simply write ϕ*ψ.
=#

function Base.:*(ϕ::Factor, ψ::Factor)
    ϕnames = variablenames(ϕ)
    ψnames = variablenames(ψ)
    ψonly = setdiff(ψ.vars, ϕ.vars)
    table = FactorTable()
    for (ϕa,ϕp) in ϕ.table
        for a in assignments(ψonly)
            a = merge(ϕa, a)
            ψa = select(a, ψnames)
            table[a] = ϕp * get(ψ.table, ψa, 0.0)
        end
    end
    vars = vcat(ϕ.vars, ψonly)
    return Factor(vars, table)
end

#= Algorithm 3.2. A method for
marginalizing a variable named
name from a factor ϕ.
=#

function marginalize(ϕ::Factor, name)
    table = FactorTable()
    for (a, p) in ϕ.table
        a′ = delete!(copy(a), name)
        table[a′] = get(table, a′, 0.0) + p
    end
    vars = filter(v -> v.name != name, ϕ.vars)
    return Factor(vars, table)
end

#= Algorithm 3.3. Two methods for factor conditioning given some evidence.
The first takes a factor ϕ and returns a new factor whose table
entries are consistent with the variable named name having the value
value. The second takes a factor ϕ and applies evidence in the form
of a named tuple. The in_scope method returns true if the variable named name 
is within the scope of the factor ϕ.
=#

in_scope(name, ϕ) = any(name == v.name for v in ϕ.vars)
function condition(ϕ::Factor, name, value)
    if !in_scope(name, ϕ)
        return ϕ
    end
    table = FactorTable()
    for (a, p) in ϕ.table
        if a[name] == value
            table[delete!(copy(a), name)] = p
        end
    end
    vars = filter(v -> v.name != name, ϕ.vars)
    return Factor(vars, table)
end
function condition(ϕ::Factor, evidence)
    for (name, value) in pairs(evidence)
        ϕ = condition(ϕ, name, value)
    end
    return ϕ
end

#= Algorithm 3.4. A naive exact inference algorithm for a discrete
Bayesian network bn, which takes
as input a set of query variable
names query and evidence associating values with observed variables. The algorithm computes a
joint distribution over the query
variables in the form of a factor.
We introduce the ExactInference
type to allow infer to be called
with different inference methods,
as shall be seen in the rest of this chapter.
=#

struct ExactInference end
function infer(M::ExactInference, bn, query, evidence)
    ϕ = prod(bn.factors)
    ϕ = condition(ϕ, evidence)
    for name in setdiff(variablenames(ϕ), query)
        ϕ = marginalize(ϕ, name)
    end
    return normalize!(ϕ)
end