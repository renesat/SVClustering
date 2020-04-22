import OSQP;
import Random.bitrand;
using Distributed: @distributed;
using Query
import MLJBase;

export PMSSVC

mutable struct PMSSVC <: MLJBase.Unsupervised
    spheres::Int64
    v::Float64
    kernel::Function
    nCheckPoints::Int64
end

PMSSVC(;spheres = 1, v = 1.0,  kernel=dot, nCheckPoints = 50) = PMSSVC(
    spheres, v, kernel, nCheckPoints
)

function MLJBase.fit(model::PMSSVC, verbosity, inX; optimizer = () -> OSQP.Optimizer(verbose = 0))
    local X = MLJBase.matrix(inX)
    local N = size(X)[1]

    local U = zeros(N, model.spheres)
    for i in 1:N
        j = rand(1:model.spheres)
        U[i, j] = 1
    end

    local U_prev = nothing
    local βspheres = nothing
    local calcs = nothing
    while U_prev != U
        U_prev = U
        calcs = @distributed vcat for j in 1:model.spheres
            Xᵢ = X[filter(i -> U[i, j] == 1, 1:N), :]
            if isempty(Xᵢ)
                []
            else
                βValues, _ = solveSVDDProblem(Xᵢ, model.kernel, model.v; optimizer = optimizer)

                calc = SVDDDistCalculator(
                    β = βValues,
                    X = Xᵢ,
                    kernel = model.kernel,
                    nCheckPoints = model.nCheckPoints
                )
                [calc]
            end
        end

        # Upadate U
        U = zeros(N, model.spheres)
        for i in 1:N
            local x = X[i, :]
            _, minj = map(
                j -> calculateR(calcs[j], x)^2 - calcs[j].R^2,
                1:length(calcs)
            ) |> findmin
            U[i, minj] = 1
        end
    end


    # Get graph
    g = SimpleGraph(N)
    for i in 1:N-1
        for j in i+1:N
            if any([isLineInSphere(calcs[a], X[i, :], X[j, :])
                    for a in 1:length(calcs)])
                add_edge!(g, i, j);
            end
        end
    end

    # Result
    fitresult = (
        βs = βspheres,
        calcs = calcs,
        graph = g,
        X = X
    )
    cache=nothing
    report=nothing
    return fitresult, cache, report
end

function MLJBase.predict(::PMSSVC, fitresult, Xnew)
    X₁ = MLJBase.matrix(Xnew)
    N₁ = size(X₁)[1]

    X₂ = fitresult.X
    N₂ = size(X₂)[1]

    X = vcat(X₂, X₁)
    N = size(X)[1]


    spheres = length(fitresult.calcs)
    g = copy(fitresult.graph)

    add_vertices!(g, N₁)
    for i in N₂+1:N-1
        for j in vcat(i+1:N, 1:N₂)
            if any([isLineInSphere(fitresult.calcs[a], X[i, :], X[j, :])
                    for a in 1:spheres])
                add_edge!(g, i, j);
            end
        end
    end

    [filter(t -> t > 0, x .-  N₂) for x in (g |> connected_components)]
end
