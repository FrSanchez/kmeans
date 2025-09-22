#pragma once

#include <vector>

class KMeans {
public:
    using Matrix = std::vector<std::vector<float>>;
    using Vector = std::vector<float>;

    static Matrix select_initial_centroids(int seed, int k, const Matrix& data);

    // New methods
    static std::vector<int> assign_points(const Matrix& data, const Matrix& centroids);
    static Matrix update_centroids(const Matrix& data, const std::vector<int>& labels, const Matrix& old_centroids);
    static float compute_objective(const Matrix& data, const std::vector<int>& labels, const Matrix& centroids);
    static void kmeans(const Matrix& data, Matrix centroids, int max_iters, float tol,
                       Matrix& final_centroids, std::vector<int>& final_labels, float& loss);
};