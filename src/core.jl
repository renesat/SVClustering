import JuMP;
using OSQP;

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

    βValue = β .|> value .|> copy

    βValue, model
end

