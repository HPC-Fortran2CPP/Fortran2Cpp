#include <iostream>
#include <omp.h>

int main() {
    int x = 2;

    #pragma omp task shared(x) mergeable
    {
        x = x + 1;
    }

    std::cout << "x = " << x << std::endl;

    return 0;
}
