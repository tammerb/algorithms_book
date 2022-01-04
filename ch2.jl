# A variable is given a name (symbol) and int
struct Variable
    name::Symbol
    r::Int # number of possible values
end

# Assignment is mapping from variable names to values (ints)
const Assignment = Dict{Symbol,Int}

const FactorTable = Dict{Assignment,Float64}

# A factors is defined by a factor table
struct Factor
    vars::Vector{Variable}
    table::FactorTable
end

### # Utility functions:

# Returns variables names associated with a factor
variablenames(ϕ::Factor) = [var.name for var in ϕ.vars]

# Selects a subset of an assignment
select(a::Assignment, varnames::Vector{Symbol}) =
    Assignment(n=>a[n] for n in varnames)

    # Enumerates possible assignments
function assignments(vars::AbstractVector{Variable})
    names = [var.name for var in vars]
    return vec([Assignment(n=>v for (n,v) in zip(names, values))
    for values in product((1:v.r for v in vars)...)])
end

# Normalizes factors
function normalize!(ϕ::Factor)
    z = sum(p for (a,p) in ϕ.table)
    for (a,p) in ϕ.table
        ϕ.table[a] = p/z
    end
    return ϕ
end

# A discrete BN in terms of set of vars, factors, and graph
struct BayesianNetwork
    vars::Vector{Variable}
    factors::Vector{Factor}
    graph::SimpleDiGraph{Int64}
end

# Evaluates the prob of an assignement given a BN
function probability(bn::BayesianNetwork, assignment)
    subassignment(ϕ) = select(assignment, variablenames(ϕ))
    probability(ϕ) = get(ϕ.table, subassignment(ϕ), 0.0)
    return prod(probability(ϕ) for ϕ in bn.factors)
end


