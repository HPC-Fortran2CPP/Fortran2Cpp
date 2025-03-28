#include <iostream>
#include <omp.h>

int main() {
    int a = 0;
    int i;

    #pragma omp parallel shared(a) private(i)
    {
        #pragma omp master
        {
            a = 0;
        }

        #pragma omp barrier

        #pragma omp for reduction(+:a)
        for (i = 1; i <= 10; ++i) {
            a = a + i;
        }

        #pragma omp single
        {
            std::cout << "Sum is " << a << std::endl;
        }
    }

    return 0;
}
