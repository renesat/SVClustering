using Distances: pairwise, Euclidean;
import MLJBase;


export compactnessMeasures, dunnIndex, daviesBouldinIndex

@doc raw"""
    compactnessMeasures(X, clusters)
"""
function compactnessMeasures(X::AbstractMatrix, clusters)
    local N = size(X, 1)

    measures = zeros(Float64, length(clusters))
    @simd for k in 1:length(clusters)
        @inbounds measures[k] = sum(pairwise(Euclidean(), X[clusters[k], :], dims = 1))
        @inbounds measures[k] *= 2
        @inbounds measures[k] /= (length(clusters[k]) - 1)
    end
    (measures |> sum) / N
end

function interDistance(cluster₁, cluster₂)
    local distX = pairwise(Euclidean(), cluster₁, cluster₂, dims = 1)
    local N₁ = size(cluster₁, 1)
    local N₂ = size(cluster₂, 1)

    sum(distX) / N₁ / N₂
end

function intraDistance(cluster)
    local distX = pairwise(Euclidean(), cluster, dims = 1)
    local N = size(cluster, 1)

    sum(distX) / N / (N - 1)  * 2
end


@doc raw"""
    dunnIndex(X, clusters)
"""
function dunnIndex(X::AbstractMatrix, clusters)
    local distX = pairwise(Euclidean(), X, dims = 1)
    local N = size(X, 1)

    local interMin = Inf;
    for i in 1:length(clusters)
        for j in i+1:length(clusters)
            interMin= min(
                interMin,
                interDistance(
                    X[clusters[i], :],
                    X[clusters[j], :]
                )
            )
        end
    end

    intraMax = max([
        intraDistance(X[cluster, :])
        for cluster in clusters
    ]...)

    interMin / intraMax
end

@doc raw"""
    daviesBouldinIndex(X, clusters)
"""
function daviesBouldinIndex(inX, clusters)
    local X = MLJBase.matrix(inX)
    local distX = pairwise(Euclidean(), X, dims = 1)
    local N = size(X)[1]

    index = sum([
        max(0, [
            (intraDistance(X[cluster₁, :]) + intraDistance(X[cluster₂, :])) / interDistance(
                X[cluster₁, :],
                X[cluster₂, :]
            )
            for cluster₂ in clusters
            if cluster₁ != cluster₂
        ]...)
        for cluster₁ in clusters
    ]) / length(clusters)
    index
end
