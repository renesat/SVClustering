import MLJBase;
using LinearAlgebra: dot;
using LightGraphs;
using JuMP;
using OSQP;


export SVDDDistCalculator, calculateR, SVDD

@doc raw"""
    SVDDDistCalculator(R, β, X, kernel, nCheckPoints)
    SVDDDistCalculator(;β, X, kernel, nCheckPoints = 20, R = nothing)
"""
struct SVDDDistCalculator
    R::Float64
    β::AbstractArray
    X::AbstractMatrix
    kernel::Function
    nCheckPoints::Int64

    R₃::Float64

    function SVDDDistCalculator(R, β, X, kernel, nCheckPoints)
        local R₃ = calculateR₃(β, X, kernel)
        new(R, β, X, kernel, nCheckPoints, R₃)
    end

    function SVDDDistCalculator(;R = nothing, β, X, kernel, nCheckPoints = 20)

        local R₃ = calculateR₃(β, X, kernel)

        if isnothing(R)
            local N = length(β)
            R = sum([calculateR(β, X, kernel, R₃, X[i, :])
                     for i in 1:N])/N
        end

        new(
            R, β, X, kernel, nCheckPoints, R₃
        )
    end

end

@doc raw"""
    calculateR₃(β, X, kernel)

Is support function for calculate $\sum_i \sum_j \beta_i \beta_j K(x_i, x_j)$.
"""
function calculateR₃(β, X, kernel)
    sum([β[i] * β[j] * kernel(X[i,:], X[j,:])
         for i in 1:length(β) for j in 1:length(β)])
end

@doc raw"""
    calculateR(β, X, kernel, x)
    calculateR(β, X, kernel, R₃, x)
    calculateR(calc::SVDDDistCalculator, x)
"""
function calculateR(β, X, kernel, R₃, x)
    N = length(β)
    R₁ = kernel(x, x)
    R₂ = -2 * sum([β[i] * kernel(X[i,:], x) for i in 1:N])
    R₃ = R₃
    sqrt(R₁ + R₂ + R₃)
end

calculateR(β, X, kernel, x) = calculateR(
    β, X, kernel,
    calculateR₃(β, X, kernel),
    x
)

calculateR(calc::SVDDDistCalculator, x) = calculateR(
    calc.β, calc.X, calc.kernel, calc.R₃,
    x
)

@doc raw"""
    isLineInSphere(calc::SVDDDistCalculator, x₁, x₂)

If `cal.nCheckPoints` points between `x₁` and `x₂` on one side of sphere
($R(x) > R$) is true.
"""
function isLineInSphere(calc::SVDDDistCalculator, x₁, x₂)
    isSphere = true
    for i in 1:calc.nCheckPoints+1
        α = (i-1)/calc.nCheckPoints
        xᵢ = x₁ * α + x₂ * (1 - α)
        if calculateR(calc, xᵢ) > calc.R
            isSphere = false
            break
        end
    end
    isSphere
end

"""
    SVDD(v, kernel, nCheckPoints)
    SVDD(;v = 1.0,  kernel=dot, nCheckPoints = 50)

It structure implement Support Vector Clustering algorythm by
[Ben-Hur, Asa, David Horn, Hava T. Siegelmann, и Vladimir Vapnik. «Support Vector Clustering»](http://www.jmlr.org/papers/v2/horn01a.html).

Model inplemented to MLJ.jl.
"""
mutable struct SVDD <: MLJBase.Unsupervised
    v::Float64
    kernel::Function
    nCheckPoints::Int64
end

SVDD(;v = 1.0,  kernel=dot, nCheckPoints = 50) = SVDD(v, kernel, nCheckPoints)

function MLJBase.fit(model::SVDD, verbosity, X; optimizer = OSQP.Optimizer)
    df = MLJBase.matrix(X)
    N = size(df)[1]

    βValue, _ = solveSVDDProblem(df, model.kernel, model.v; optimizer = optimizer)

    calc = SVDDDistCalculator(
        β = βValue,
        X = df,
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


    # Result
    fitresult = (
        β = βValue,
        calc = calc,
        graph = g,
        X = df
    )
    cache=nothing
    report=nothing
    return fitresult, cache, report
end

function MLJBase.predict(::SVDD, fitresult, Xnew)
    df₁ = MLJBase.matrix(Xnew)
    N₁ = size(df₁)[1]

    df₂ = fitresult.X
    N₂ = size(df₂)[1]

    df = vcat(df₂, df₁)
    N = size(df)[1]


    g = copy(fitresult.graph)

    add_vertices!(g, N₁)
    for i in N₂+1:N-1
        for j in vcat(i+1:N, 1:N₂)
            if isLineInSphere(fitresult.calc, df[i, :], df[j, :])
                add_edge!(g, i, j);
            end
        end
    end

    [filter(t -> t > 0, x .-  N₂) for x in (g |> connected_components)]
end
