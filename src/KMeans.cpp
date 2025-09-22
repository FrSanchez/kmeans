#include "KMeans.h"
#include <cmath>
#include <limits>
#include <unordered_map>
#include <random>

KMeans::Matrix KMeans::select_initial_centroids(int seed, int k, const Matrix& data) {
    if (k > data.size()) throw std::invalid_argument("k cannot be greater than the number of data points");

    std::vector<size_t> indices(data.size());
    for (size_t i = 0; i < data.size(); ++i) indices[i] = i;

    std::mt19937 gen(seed);
    std::shuffle(indices.begin(), indices.end(), gen);

    Matrix centroids;
    for (int i = 0; i < k; ++i) {
        centroids.push_back(data[indices[i]]);
    }

    return centroids;
}


// Helper: Euclidean distance squared between two vectors
static float euclidean_dist_sq(const std::vector<float>& a, const std::vector<float>& b) {
    float acc = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        float d = a[i] - b[i];
        acc += d * d;
    }
    return acc;
}

// assign_points: assigns each point to the nearest centroid
std::vector<int> KMeans::assign_points(const Matrix& data, const Matrix& centroids) {
    size_t n = data.size();
    size_t k = centroids.size();
    std::vector<int> labels(n);
    for (size_t idx = 0; idx < n; ++idx) {
        int best_i = -1;
        float best_dist_sq = std::numeric_limits<float>::max();
        for (size_t i = 0; i < k; ++i) {
            float dist_sq = euclidean_dist_sq(data[idx], centroids[i]);
            if (dist_sq < best_dist_sq) {
                best_dist_sq = dist_sq;
                best_i = static_cast<int>(i);
            }
        }
        labels[idx] = best_i;
    }
    return labels;
}

// update_centroids: computes centroids as mean of assigned points
KMeans::Matrix KMeans::update_centroids(const Matrix& data, const std::vector<int>& labels, const Matrix& old_centroids) {
    size_t k = old_centroids.size();
    size_t dim = old_centroids[0].size();
    Matrix new_centroids(k, std::vector<float>(dim, 0.0f));
    std::vector<int> counts(k, 0);

    for (size_t idx = 0; idx < data.size(); ++idx) {
        int label = labels[idx];
        counts[label]++;
        for (size_t j = 0; j < dim; ++j) {
            new_centroids[label][j] += data[idx][j];
        }
    }
    for (size_t i = 0; i < k; ++i) {
        if (counts[i] > 0) {
            for (size_t j = 0; j < dim; ++j) {
                new_centroids[i][j] /= counts[i];
            }
        } else {
            new_centroids[i] = old_centroids[i]; // keep old centroid if empty cluster
        }
    }
    return new_centroids;
}

// compute_objective: sum of squared distances from points to their centroids
float KMeans::compute_objective(const Matrix& data, const std::vector<int>& labels, const Matrix& centroids) {
    float total = 0.0f;
    for (size_t idx = 0; idx < data.size(); ++idx) {
        int label = labels[idx];
        total += euclidean_dist_sq(data[idx], centroids[label]);
    }
    return total;
}

// kmeans: the main iterative algorithm
void KMeans::kmeans(const Matrix& data, Matrix centroids, int max_iters, float tol,
                    Matrix& final_centroids, std::vector<int>& final_labels, float& loss)
{
    size_t k = centroids.size();
    size_t dim = (k > 0) ? centroids[0].size() : 0;

    for (int iter = 0; iter < max_iters; ++iter) {
        // Step A: assign labels
        std::vector<int> labels = assign_points(data, centroids);

        // Step B: update centroids
        Matrix new_centroids = update_centroids(data, labels, centroids);

        // Step C: check convergence
        bool converged = true;
        for (size_t i = 0; i < k; ++i) {
            float move = std::sqrt(euclidean_dist_sq(new_centroids[i], centroids[i]));
            if (move >= tol) {
                converged = false;
                break;
            }
        }

        centroids = new_centroids;
        if (converged) {
            break;
        }
    }
    // Final labels and loss calculation
    final_labels = assign_points(data, centroids);
    loss = compute_objective(data, final_labels, centroids);
    final_centroids = centroids;
}