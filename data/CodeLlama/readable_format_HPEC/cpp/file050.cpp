#include <iostream>
#include <vector>
#include <omp.h>
#include <cstdlib>
#include <cstring>

int main(int argc, char** argv) {
    int len = 1000;
    int n, m;
    std::vector<std::string> args(argc);
    std::vector<std::vector<float>> b;

    // Check command line arguments
    if (argc == 0) {
        std::cout << "No command line arguments provided." << std::endl;
        return 1;
    }

    // Parse command line arguments
    for (int ix = 0; ix < argc; ++ix) {
        args[ix] = argv[ix];
    }

    // Check if length is provided
    if (argc >= 1) {
        char* endptr = nullptr;
        len = std::strtol(args[0].c_str(), &endptr, 10);
        if (*endptr != ' ') {
            std::cout << "Error, invalid integer value." << std::endl;
            return 1;
        }
    }

    n = len;
    m = len;

    // Allocate and initialize b
    b.resize(n, std::vector<float>(m, 0.0f));

    // Parallel loop to fill the second dimension
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; ++i) {
        for (int j = 1; j < m; ++j) {
            b[i][j] = b[i][j - 1];
        }
    }

    // Printing b(5,5) (assuming 0-based indexing in C++)
    std::cout << "b(5,5) = " << b[4][4] << std::endl; // Adjusted for 0-based indexing

    return 0;
}
