#include <iostream>
#include <omp.h>

int main() {
    int x = 0;
    int i;

    #pragma omp parallel for ordered
    for (int i = 1; i <= 100; i++) {
        #pragma omp ordered
        x++;
    }

    std::cout << "x = " << x << std::endl;

    return 0;
}
