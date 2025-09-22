#pragma once

#include <string>
#include <vector>

class NpyLoader {
public:
    // Loads a flat float array from .npy
    static std::vector<float> loadFloatArray(const std::string& filename);

    // Loads a 2D float array from .npy
    static std::vector<std::vector<float>> loadFloatMatrix(const std::string& filename);
};