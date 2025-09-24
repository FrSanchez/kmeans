#include "KMeans.h"
#include <xtensor/views/xview.hpp>
#include <xtensor/containers/xadapt.hpp>
#include <xtensor/views/xbroadcast.hpp>
#include <xtensor/core/xmath.hpp> // for xt::sum, xt::sqrt
#include <xtensor/io/xio.hpp>
#include <random>
#include <algorithm>
#include <stdexcept>

// Select k random rows from the data matrix as initial centroids
KMeans::Matrix KMeans::select_initial_centroids(int seed, int k, const Matrix& data) {
    size_t n = data.shape()[0];
    if (k > n) throw std::invalid_argument("k cannot be greater than the number of data points");
    std::vector<size_t> indices(n);
    for (size_t i = 0; i < n; ++i) indices[i] = i;
    std::mt19937 gen(seed);
    std::shuffle(indices.begin(), indices.end(), gen);

    // Stack selected rows
     std::vector<xt::xarray<double>> rows;
    for (int i = 0; i < k; ++i) {
        rows.push_back(xt::view(data, indices[i], xt::all()));
    }
    return xt::stack(xt::xtuple(rows), 0);
}

// Assign each point to the nearest centroid
std::vector<int> KMeans::assign_points(const Matrix& data, const Matrix& centroids) {
    size_t n = data.shape()[0];
    size_t k = centroids.shape()[0];
    std::vector<int> labels(n, 0);

    for (size_t idx = 0; idx < n; ++idx) {
        auto point = xt::view(data, idx, xt::all());
        double min_dist = std::numeric_limits<double>::max();
        int min_idx = 0;
        for (size_t c = 0; c < k; ++c) {
            auto centroid = xt::view(centroids, c, xt::all());
            double dist = xt::sum(xt::square(point - centroid))();
            if (dist < min_dist) {
                min_dist = dist;
                min_idx = static_cast<int>(c);
            }
        }
        labels[idx] = min_idx;
    }
    return labels;
}

// Update the centroids as the mean of assigned data points
KMeans::Matrix KMeans::update_centroids(const Matrix& data, const std::vector<int>& labels, const Matrix& old_centroids) {
    size_t k = old_centroids.shape()[0];
    size_t dim = old_centroids.shape()[1];
    Matrix new_centroids = xt::zeros<double>({k, dim});
    std::vector<size_t> counts(k, 0);

    size_t n = data.shape()[0];
    for (size_t i = 0; i < n; ++i) {
        int label = labels[i];
        new_centroids(label, xt::all()) += xt::view(data, i, xt::all());
        counts[label]++;
    }

    for (size_t c = 0; c < k; ++c) {
        if (counts[c] > 0) {
            new_centroids(c, xt::all()) /= static_cast<double>(counts[c]);
        } else {
            // If no points assigned, keep old centroid
            new_centroids(c, xt::all()) = xt::view(old_centroids, c, xt::all());
        }
    }
    return new_centroids;
}

// Compute the sum of squared distances (objective)
double KMeans::compute_objective(const Matrix& data, const std::vector<int>& labels, const Matrix& centroids) {
    double total = 0.0;
    size_t n = data.shape()[0];
    for (size_t i = 0; i < n; ++i) {
        auto point = xt::view(data, i, xt::all());
        auto centroid = xt::view(centroids, labels[i], xt::all());
        total += xt::sum(xt::square(point - centroid))();
    }
    return total;
}

// The kmeans main loop
void KMeans::kmeans(const Matrix& data, Matrix centroids, int max_iters, double tol,
                    Matrix& final_centroids, std::vector<int>& final_labels, double& loss) {
    size_t k = centroids.shape()[0];
    for (int iter = 0; iter < max_iters; ++iter) {
        std::vector<int> labels = assign_points(data, centroids);
        Matrix new_centroids = update_centroids(data, labels, centroids);
        // Check for convergence
        bool converged = true;
        for (size_t c = 0; c < k; ++c) {
            double move = xt::sqrt(xt::sum(xt::square(xt::view(new_centroids, c, xt::all()) - xt::view(centroids, c