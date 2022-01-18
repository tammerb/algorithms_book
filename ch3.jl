include("ch2.jl")

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

#=
Algorithm 3.5. An implementation of the sum-product variable
elimination algorithm, which takes
in a Bayesian network bn, a list
of query variables query, and evidence evidence. The variables are
processed in the order given by
ordering.
=#

struct VariableElimination
    ordering # array of variable indices
end
function infer(M::VariableElimination, bn, query, evidence)
    Φ = [condition(ϕ, evidence) for ϕ in bn.factors]
    for i in M.ordering
        name = bn.vars[i].name
        if name ∉ query
            inds = findall(ϕ->in_scope(name, ϕ), Φ)
            if !isempty(inds)
                ϕ = prod(Φ[inds])
                deleteat!(Φ, inds)
                ϕ = marginalize(ϕ, name)
                push!(Φ, ϕ)
            end
        end
    end
    return normalize!(prod(Φ))
end

#=
Algorithm 3.6. A method for sampling 
an assignment from a Bayesian  network 
bn. We also provide a method for 
sampling an assignment from a factor ϕ.
=#
function Base.rand(ϕ::Factor)
    tot, p, w = 0.0, rand(), sum(values(ϕ.table))
    for (a,v) in ϕ.table
        tot += v/w
        if tot >= p
            return a
        end
    end
    return Assignment()
end
function Base.rand(bn::BayesianNetwork)
    a = Assignment()
    for i in topological_sort_by_dfs(bn.graph)
        name, ϕ = bn.vars[i].name, bn.factors[i]
        a[name] = rand(condition(ϕ, a))[name]
    end
    return a
end

#=
Algorithm 3.7. The direct sampling inference method, which takes
a Bayesian network bn, a list of query variables query, and
evidence evidence. The method draws m samples from the Bayesian
network and retains those samples that are consistent with the
evidence. A factor over the query variables is 
returned. This method can fail if no samples that satisfy the 
evidence are found.=#

struct DirectSampling
    m # number of samples
end
function infer(M::DirectSampling, bn, query, evidence)
    table = FactorTable()
    for i in 1:(M.m)
        a = rand(bn)
        if all(a[k] == v for (k,v) in pairs(evidence))
            b = select(a, query)
            table[b] = get(table, b, 0) + 1
        end
    end
    vars = filter(v->v.name ∈ query, bn.vars)
    return normalize!(Factor(vars, table))
end

#=
Algorithm 3.8. The likelihood
weighted sampling inference
method, which takes a Bayesian
network bn, a list of query
variables query, and evidence
evidence. The method draws
m samples from the Bayesian
network but sets values from
evidence when possible, keeping
track of the conditional probability
when doing so. These probabilities
are used to weight the samples
such that the final inference
estimate is accurate. A factor over
the query variables is returned.
=#

struct LikelihoodWeightedSampling
    m # number of samples
end
function infer(M::LikelihoodWeightedSampling, bn, query, evidence)
    table = FactorTable()
    ordering = topological_sort_by_dfs(bn.graph)
    for i in 1:(M.m)
        a, w = Assignment(), 1.0
        for j in ordering
            name, ϕ = bn.vars[j].name, bn.factors[j]
            if haskey(evidence, name)
                a[name] = evidence[name]
                w *= ϕ.table[select(a, variablenames(ϕ))]
            else
                a[name] = rand(condition(ϕ, a))[name]
            end
        end
        b = select(a, query)
        table[b] = get(table, b, 0) + w
    end
    vars = filter(v->v.name ∈ query, bn.vars)
    return normalize!(Factor(vars, table))
end

#=
Algorithm 3.9. A method for obtaining P(Xi
| x−i) for a Bayesian
network bn given a current assignment a.
=#

function blanket(bn, a, i)
    name = bn.vars[i].name
    val = a[name]
    a = delete!(copy(a), name)
    Φ = filter(ϕ -> in_scope(name, ϕ), bn.factors)
    ϕ = prod(condition(ϕ, a) for ϕ in Φ)
    return normalize!(ϕ)
end

#=
Algorithm 3.10. Gibbs sampling implemented for a Bayesian network bn with evidence evidence
and an ordering ordering. The method iteratively updates the assignment a for m iterations.
=#

function update_gibbs_sample!(a, bn, evidence, ordering)
    for i in ordering
        name = bn.vars[i].name
        if !haskey(evidence, name)
            b = blanket(bn, a, i)
            a[name] = rand(b)[name]
        end
    end
end
function gibbs_sample!(a, bn, evidence, ordering, m)
    for j in 1:m
        update_gibbs_sample!(a, bn, evidence, ordering)
    end
end
struct GibbsSampling
    m_samples # number of samples to use
    m_burnin # number of samples to discard during burn-in
    m_skip # number of samples to skip for thinning
    ordering # array of variable indices
end
function infer(M::GibbsSampling, bn, query, evidence)
    table = FactorTable()
    a = merge(rand(bn), evidence)
    gibbs_sample!(a, bn, evidence, M.ordering, M.m_burnin)
    for i in 1:(M.m_samples)
        gibbs_sample!(a, bn, evidence, M.ordering, M.m_skip)
        b = select(a, query)
        table[b] = get(table, b, 0) + 1
    end
    vars = filter(v->v.name ∈ query, bn.vars)
    return normalize!(Factor(vars, table))
end

#=
Algorithm 3.11. Inference in a multivariate Gaussian distribution D.
A vector of integers specifies the
query variables in the query argument, and a vector of integers
specifies the evidence variables
in the evidencevars argument.
The values of the evidence variables are contained in the vector
evidence. The Distributions.jl
package defines the MvNormal distribution.
=#

function infer(D::MvNormal, query, evidencevars, evidence)
    μ, Σ = D.μ, D.Σ.mat
    b, μa, μb = evidence, μ[query], μ[evidencevars]
    A = Σ[query,query]
    B = Σ[evidencevars,evidencevars]
    C = Σ[query,evidencevars]
    μ = μ[query] + C * (B\(b - μb))
    Σ = A - C * (B \ C')
    return MvNormal(μ, Σ)
end