#include <iostream>
#include <omp.h>

int main() {
    int numThreads = 0;

    #pragma omp parallel
    {
        if (omp_get_thread_num() == 0) {
            numThreads = omp_get_num_threads();
        } else {
            std::cout << "numThreads = " << numThreads << std::endl;
        }
    }

    return 0;
}