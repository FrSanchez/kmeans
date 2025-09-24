#pragma once

#include <string>
#include <vector>

class NpyLoader {
public:
    // Loads a flat float array from .npy
    static std::vector<double> loadFloatArray(const std::string& filename);

    // Loads a 2D float array from .npy
    static std::vector<std::vector<double>> loadFloatMatrix(const std::string& filename);
};