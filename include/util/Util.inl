#pragma once

#include <filesystem>
#include <fstream>
#include <iostream>

namespace Util {

bool doesFileExist(const std::string &filepath) {
    std::ifstream f(filepath.c_str());
    return f.good();
}

void checkCudaErrorCode(cudaError_t code) {
    if (code != cudaSuccess) {
        std::string errMsg =
            "CUDA operation failed with code: " + std::to_string(code) + " (" +
            cudaGetErrorName(code) +
            "), with message: " + cudaGetErrorString(code);
        std::cerr << errMsg << '\n';
        throw std::runtime_error(errMsg);
    }
}

std::vector<std::string> getFilesInDirectory(const std::string &dirPath) {
    std::vector<std::string> fileNames;
    for (const auto &entry : std::filesystem::directory_iterator(dirPath)) {
        if (entry.is_regular_file()) {
            fileNames.push_back(entry.path().string());
        }
    }
    return fileNames;
}
} // namespace Util