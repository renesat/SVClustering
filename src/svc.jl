import MLJBase;
using LinearAlgebra: dot;
using LightGraphs;
using JuMP;
using OSQP;
using Distances;
using Statistics;


export SVC

"""
    SVC(v, kernel, nCheckPoints)
    SVC(;v = 1.0,  kernel=dot, nCheckPoints = 50)

It structure implement Support Vector Clustering algorythm by
[Ben-Hur, Asa, David Horn, Hava T. Siegelmann, и Vladimir Vapnik. «Support Vector Clustering»](http://www.jmlr.org/papers/v2/horn01a.html).

Model inplemented to MLJ.jl.
"""
mutable struct SVC <: MLJBase.Unsupervised
    v::Float64
    kernel::Function
    nCheckPoints::Int64
end

SVC(;v = 1.0,  kernel=dot, nCheckPoints = 50) = SVC(v, kernel, nCheckPoints)

function MLJBase.fit(model::SVC, verbosity, X; optimizer = () -> Ipopt.Optimizer(print_level = 0))#OSQP.Optimizer(verbose = 0))
    df = MLJBase.matrix(X)
    N = size(df)[1]

    # βValue, _ = solveSVDDProblem(df, model.kernel, model.v; optimizer = optimizer)
    βValue = solveSVDDProblem(df, model.kernel, model.v; optimizer = optimizer)

    isSVPoints = βValue .> 10^-4
    if length(isSVPoints) == 0
        return nothing, nothing, nothing
    end

    calc = SVDDDistCalculator(
        β = βValue[isSVPoints],
        X = df[isSVPoints, :],
        kernel = model.kernel,
        nCheckPoints = model.nCheckPoints
    )

    # Get graph
    g = SimpleGraph(N)
    for i in 1:N-1
        for j in i+1:N
            if isLineInSphere(calc, df[i, :], df[j, :])
                add_edge!(g, i, j);
            end
        end
    end

    clusters =  g |> connected_components |> a -> filter(x -> length(x) > 1, a)
    clusteringPoints = vcat(clusters...)
    for i in 1:N
        if all(i .!= clusteringPoints)
            _, j = findmin([Euclidean()(df[i, :], mean(df[cluster, :], dims=1)) for cluster in clusters])
            push!(clusters[j], i)
        end
    end


    # Result
    fitresult = (
        β = βValue,
        clusters = clusters,
        calc = calc,
        graph = g,
        X = df
    )
    cache=nothing
    report=nothing
    return fitresult, cache, report
end
