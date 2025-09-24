#include <iostream>
#include <string>
#include <xtensor/containers/xarray.hpp>
#include <xtensor/io/xnpy.hpp>
#include <xtensor/io/xio.hpp>
#include "KMeans.h"


int main(int argc, char* argv[]) {
    int k = 5;
    int seed = 0;
    std::string input_file;

    // Parse command-line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "-k") && (i + 1 < argc)) {
            k = std::stoi(argv[++i]);
        } else if ((arg == "-s" || arg == "--seed") && (i + 1 < argc)) {
            seed = std::stoi(argv[++i]);
        } else if ((arg == "-i" || arg == "--input") && (i + 1 < argc)) {
            input_file = argv[++i];
        }
    }

    if (input_file.empty()) {
        std::cerr << "Error: Input file must be specified using -i or --input.\n";
        std::cerr << "Usage: " << argv[0] << " -i <input.npy> [-k <num_clusters>] [-s <seed>]\n";
        return 1;
    }

    // Load the numpy matrix
    // auto mat = NpyLoader::loadFloatMatrix(input_file);
    auto mat = xt::load_npy<double>(input_file);
    std::cout << "Loaded matrix with shape: ";
    for (auto s : mat.shape()) std::cout << s << " ";
    std::cout << std::endl;

    // Print the matrix (xtensor supports fancy printing!)
    std::cout << mat << std::endl;
    if (k > mat.size()) {
        std::cerr << "Error: Number of clusters (k) cannot exceed number of data points.\n";
        return 1;
    }
    // std::cout << "Loaded " << mat.size() << " rows with " << mat.size() << " columns.\n";

    // // Select initial centroids
    auto centroids = KMeans::select_initial_centroids(seed, k, mat);
    // std::cout << "Initial centroids selected with seed " << seed << ":\n";
    // std::cout << "Selected " << centroids.size() << " initial centroids:\n";
    // for (const auto& centroid : centroids) {
    //     for (float val : centroid) {
    //         std::cout << val << " ";
    //     }
    //     std::cout << "\n";
    // }

    // KMeans::Matrix final_centroids;
    // std::vector<int> final_labels;
    // float loss = 0.0f;
    // KMeans::kmeans(mat, centroids, 1000, 1e-8f, final_centroids, final_labels, loss);

    // std::cout << "Final loss: " << loss << "\n";
    // std::cout << "Final centroids:\n";
    // for (const auto& centroid : final_centroids) {
    //     for (float val : centroid) {
    //         std::cout << val << " ";
    //     }
    //     std::cout << "\n";
    // }

    return 0;
}