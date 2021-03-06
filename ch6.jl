include("ch5.jl")

#= Algorithm 6.1 A simple problem as a decision network.
=#

struct SimpleProblem
    bn::BayesianNetwork
    chance_vars::Vector{Variable}
    decision_vars::Vector{Variable}
    utility_vars::Vector{Variable}
    utilities::Dict{Symbol, Vector{Float64}}
end
function solve(ð«::SimpleProblem, evidence, M)
    query = [var.name for var in ð«.utility_vars]
    U(a) = sum(ð«.utilities[uname][a[uname]] for uname in query)
    best = (a=nothing, u=-Inf)
    for assignment in assignments(ð«.decision_vars)
        evidence = merge(evidence, assignment)
        Ï = infer(M, ð«.bn, query, evidence)
        u = sum(p*U(a) for (a, p) in Ï.table)
        if u > best.u
            best = (a=assignment, u=u)
        end
    end
    return best
end

#= Algorithm 6.2 A method for decision network evaluation.
=#
function value_of_information(ð«, query, evidence, M)
    Ï = infer(M, ð«.bn, query, evidence)
    voi = -solve(ð«, evidence, M).u
    query_vars = filter(v->v.name â query, ð«.chance_vars)
    for oâ² in assignments(query_vars)
        ooâ² = merge(evidence, oâ²)
        p = Ï.table[oâ²]
        voi += p*solve(ð«, ooâ², M).u
    end
    return voi
end