import JuMP;
using OSQP;
using TimerOutputs


export SVDDDistCalculator, calculateR, SVDD

@doc raw"""
    solveSVDDProblem(X; optimizer = OSQP.Optimizer)

Solve SVDD problem to find parameters β:
\[
    \max_\beta \sum_j K(x_j, x_j) \beta_j - \sum_j \sum_i K(x_j, x_i) \beta_j \beta_i
\]
with $1 < \beta_j < \frac{1}{p N}, j = 1, \dots, N$ and $\sum_j \beta_j = 1$.
"""
function solveSVDDProblem(X, kernel, v; optimizer = OSQP.Optimizer)
    N = size(X)[1]
    C =  1 / (N * v)

    model = JuMP.Model(optimizer)

    JuMP.@variable(model, β[1:N])

    W₁ = [kernel(X[i, :], X[i, :]) * β[i] for i in 1:N]
    W₂ = [kernel(X[i, :], X[j, :]) * β[i] * β[j] for i in 1:N for j in 1:N]
    W = sum(W₁) - sum(W₂)

    JuMP.@constraint(model, β .>= 0)
    JuMP.@constraint(model, β .<= C)
    JuMP.@constraint(model, sum(β) == 1)
    JuMP.@objective(model, Max, W)

    JuMP.optimize!(model)

    β .|> value
end


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

            R = sum(
                calculateR(β, X, kernel, R₃, X[i, :])
                for i in 1:N
            ) / N
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
    sum(
        β[i] * β[j] * kernel(X[i,:], X[j,:])
        for i in 1:length(β) for j in 1:length(β)
    )
end

@doc raw"""
    calculateR(β, X, kernel, x)
    calculateR(β, X, kernel, R₃, x)
    calculateR(calc::SVDDDistCalculator, x)
"""
function calculateR(β, X, kernel, R₃, x)
    R = kernel(x, x)
    R += -2 * sum(β[i] * kernel(X[i,:], x) for i in 1:length(β))
    R += R₃
    sqrt(R)
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

function isPointInSphere(calc::SVDDDistCalculator, x)
    calculateR(calc, x) <= calc.R
end

@doc raw"""
    isLineInSphere(calc::SVDDDistCalculator, x₁, x₂)

If `cal.nCheckPoints` points between `x₁` and `x₂` on one side of sphere
($R(x) > R$) is true.
"""
function isLineInSphere(calc::SVDDDistCalculator, x₁, x₂)
    all(
        α -> isPointInSphere(
            calc,
            x₁ * α + x₂ * (1 - α)
        ),
        i/calc.nCheckPoints for i in 1:calc.nCheckPoints
    )
end
