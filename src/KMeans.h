#pragma once

#include <xtensor/containers/xarray.hpp>
#include <vector>

class KMeans {
public:
    using Matrix = xt::xarray<double>;

    static Matrix select_initial_centroids(int seed, int k, const Matrix& data);
    static std::vector<int> assign_points(const Matrix& data, const Matrix& centroids);
    static Matrix update_centroids(const Matrix& data, const std::vector<int>& labels, const Matrix& old_centroids);
    static double compute_objective(const Matrix& data, const std::vector<int>& labels, const Matrix& centroids);
    static void kmeans(const Matrix& data, Matrix centroids, int max_iters, double tol,
                       Matrix& final_centroids, std::vector<int>& final_labels, double& loss);
};
