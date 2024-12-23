#include <iostream>
#include <vector>
#include <omp.h>
#include <string>
#include <cstdlib>

int main(int argc, char** argv) {
    int len = 100;
    std::vector<std::string> args;
    std::vector<int> a;

    // Check if command line arguments are provided
    if (argc == 0) {
        std::cout << "No command line arguments provided." << std::endl;
        return 1;
    }

    // Parse command line arguments
    for (int ix = 0; ix < argc; ++ix) {
        args.push_back(argv[ix]);
    }

    // Check if at least one argument is provided
    if (argc >= 1) {
        // Attempt to convert the first argument to an integer
        char* endptr = nullptr;
        len = std::strtol(args[0].c_str(), &endptr, 10);
        if (*endptr != ' ') {
            std::cout << "Error, invalid integer value." << std::endl;
            return 1;
        }
    }

    // Allocate memory for array a
    a.resize(len);

    // Initialize array a
    for (int i = 0; i < len; ++i) {
        a[i] = i + 1; // Fortran arrays are 1-based, C++ arrays are 0-based
    }

    // Parallel loop to increment elements
    #pragma omp parallel for
    for (int i = 0; i < len - 1; ++i) {
        a[i + 1] = a[i] + 1;
    }

    // Print the value of a[50]
    std::cout << "a[50]=" << a[50] << std::endl;

    return 0;
}
