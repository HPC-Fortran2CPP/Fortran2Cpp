#include <iostream>
#include <vector>
#include <omp.h>
#include <string>
#include <cstdlib> // For std::exit

int main(int argc, char** argv) {
    int len = 100;
    int x = 10;
    std::vector<std::string> args;
    std::vector<int> a;

    // Check if command line arguments are provided
    if (argc == 0) {
        std::cout << "No command line arguments provided." << std::endl;
        return 1;
    }

    // Allocate memory for args and a based on argc
    try {
        args.resize(argc);
        a.resize(len);
    } catch (const std::bad_alloc& e) {
        std::cerr << "Allocation error, program terminated." << std::endl;
        std::exit(1);
    }

    // Get command line arguments
    for (int ix = 0; ix < argc; ++ix) {
        args[ix] = argv[ix];
    }

    // Process the first argument if provided
    if (argc >= 1) {
        char* endptr = nullptr;
        len = std::strtol(args[0].c_str(), &endptr, 10);
        if (*endptr != ' ') {
            std::cerr << "Error, invalid integer value." << std::endl;
            return 1;
        }
    }

    // Parallel loop to populate array a
    #pragma omp parallel for
    for (int i = 0; i < len; ++i) {
        a[i] = x;
        x = i + 1; // Increment x in each iteration
    }

    // Print results
    std::cout << "x=" << x << " a(0)=" << a[0] << std::endl;

    return 0;
}
