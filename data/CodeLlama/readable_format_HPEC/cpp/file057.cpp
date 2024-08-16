#include <iostream>
#include <vector>
#include <omp.h>

int main() {
    int n = 1000;
    int m = 1000;
    std::vector<std::vector<float>> b(n, std::vector<float>(m, 0.5f));

    // Parallel region for the inner loop
    #pragma omp parallel for collapse(2)
    for (int i = 1; i < n; ++i) {
        for (int j = 1; j < m; ++j) {
            b[i][j] = b[i-1][j-1];
        }
    }

    // Print the value of b(500,500)
    std::cout << "b(500,500) = " << b[499][499] << std::endl;

    return 0;
}
