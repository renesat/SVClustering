import OSQP;
import NLopt
using Ipopt

import Random.bitrand;
using Distributed: @distributed;
using Query
import MLJBase;
using Distances;
using LinearAlgebra: Symmetric

export PMSSVC

mutable struct PMSSVC <: MLJBase.Unsupervised
    spheres::Int64
    f::Float64
    k::Int64
    v::Float64
    kernel::Function
    nCheckPoints::Int64
end

PMSSVC(;spheres = 1, v = 1.0,
       kernel = dot, nCheckPoints = 50,
       f = 1.5, k = 3) = PMSSVC(
    spheres, f, k, v, kernel, nCheckPoints
)

function MLJBase.fit(model::PMSSVC, verbosity, X::AbstractMatrix; maxiter = 100, optimizer = () -> Ipopt.Optimizer(print_level = 0))#OSQP.Optimizer(verbose = 0))
    local distX = pairwise(Euclidean(), X, dims = 1)
    local N = size(X, 1)

    local minDist = hcat(distX...) |> x -> filter(a -> a > 0.0, x) |> x -> min(x...)
    local θ = model.f * minDist

    local U = [rand(1:model.spheres) for _ in 1:N]

    local U_prev = similar(U)
    local βValues = nothing
    local calcs = Union{Nothing, SVDDDistCalculator}[nothing for _ in 1:model.spheres]
    local iteration = 0
    while U_prev != U && iteration < maxiter
        iteration += 1
        U_prev = deepcopy(U)

        @simd for j in 1:model.spheres
            Xᵢ = X[U .== j, :]
            if isempty(Xᵢ)
                calcs[j] = nothing
            else
                βValues = solveSVDDProblem(Xᵢ, model.kernel, model.v; optimizer = optimizer)

                isSVPoints = βValues .> 10^-4

                if length(isSVPoints) == 0
                    calcs[j] = nothing
                else
                    calc = SVDDDistCalculator(
                        β = βValues[isSVPoints],
                        X = Xᵢ[isSVPoints, :],
                        kernel = model.kernel,
                        nCheckPoints = model.nCheckPoints
                    )
                    calcs[j] = calc
                end
            end
        end

        # Upadate U
        for i in 1:N
            local x = X[i, :]
            local ispheres = filter(
                calc -> !isnothing(calc[2]) && isPointInSphere(
                    calc[2],
                    x
                ),
                enumerate(calcs) |> collect
            )
            local goodspheres = filter(
                calc -> !isnothing(calc[2]),
                enumerate(calcs) |> collect
            )

            if length(goodspheres) == 0
                return nothing, nothing, nothing
            end

            if length(ispheres) > 0
                _, minj = map(
                    calc -> calculateR(calc[2], x)^2,
                    ispheres
                ) |> findmin
                U[i] = ispheres[minj][1]
            else
                _, minj = map(
                    calc -> calculateR(calc[2], x)^2 - calc[2].R^2,
                    goodspheres
                ) |> findmin
                U[i] = goodspheres[minj][1]
            end
        end
    end

    # Get graph


    g = SimpleGraph(N)
    local knn
    for i in 1:N-1
        knn = filter(
            x -> distX[i, x] <= θ,
            sort(1:N, by = x -> distX[i, x])[2:model.k+1]
        )
        @simd for j in i+1:N
            local isConnect = (j ∈ knn) || any(
                isLineInSphere(calcs[a], X[i, :], X[j, :])
                for a in 1:length(calcs)
                if !isnothing(calcs[a])
            )
            if isConnect
                add_edge!(g, i, j);
                add_edge!(g, j, i);
            end
        end
    end


    clusters =  g |> connected_components |> a -> filter(x -> length(x) > 1, a)
    clusteringPoints = vcat(clusters...)
    for i in 1:N
        if all(i .!= clusteringPoints)
            _, j = findmin([
                Euclidean()(X[i, :], mean(X[cluster, :], dims=1))
                for cluster in clusters
            ])
            push!(clusters[j], i)
        end
    end

    # Result
    fitresult = (
        clusters = clusters,
        calcs = calcs,
        graph = g,
        k = model.k,
        θ = θ,
        X = X
    )
    cache=nothing
    report=nothing
    return fitresult, cache, report
end
