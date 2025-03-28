#include <iostream>
#include <vector>
#include <omp.h>
#include <cstdlib>
#include <cstring>

int main(int argc, char** argv) {
    int len = 1000;
    int n, m;
    std::vector<std::string> args;
    std::vector<std::vector<float>> b;

    // Check command line arguments
    if (argc == 0) {
        std::cout << "No command line arguments provided." << std::endl;
        return 1;
    }

    // Allocate and read command line arguments
    args.resize(argc);
    for (int ix = 0; ix < argc; ++ix) {
        args[ix] = argv[ix];
    }

    // Check if first argument is an integer
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
    b.resize(n, std::vector<float>(m, 0.5f));

    // Parallel loop to fill the matrix
    #pragma omp parallel for collapse(2)
    for (int i = 1; i < n; ++i) {
        for (int j = 1; j < m; ++j) {
            b[i][j] = b[i - 1][j - 1];
        }
    }

    // Print the value of b(500,500)
    std::cout << "b(500,500) = " << b[499][499] << std::endl; // Note: C++ uses 0-based indexing

    return 0;
}
