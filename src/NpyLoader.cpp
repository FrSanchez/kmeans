#include "NpyLoader.h"
#include "cnpy.h"

std::vector<double> NpyLoader::loadFloatArray(const std::string& filename) {
    cnpy::NpyArray arr = cnpy::npy_load(filename);
    double* dataPtr = arr.data<double>();
    size_t size = arr.shape[0];
    return std::vector<double>(dataPtr, dataPtr + size);
}

std::vector<std::vector<double>> NpyLoader::loadFloatMatrix(const std::string& filename) {
    cnpy::NpyArray arr = cnpy::npy_load(filename);
    double* dataPtr = arr.data<double>();
    size_t rows = arr.shape[0];
    size_t cols = arr.shape[1];
    std::vector<std::vector<double>> matrix(rows, std::vector<double>(cols));
    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < cols; ++c) {
            matrix[r][c] = dataPtr[r * cols + c];
        }
    }
    return matrix;
}