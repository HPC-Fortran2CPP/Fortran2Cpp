program DRB132_taskdep4_orig_no_omp45
    use omp_lib
    implicit none

    !$omp parallel
    !$omp single
    call foo()
    !$omp end single
    !$omp end parallel
contains
    subroutine foo()
        implicit none
        integer :: x, y
        x = 0
        y = 2

        !$omp task depend(inout: x) shared(x)
        x = x+1                                 !!1st Child Task
        !$omp end task

        !$omp task shared(y)
        y = y-1                                 !!2nd child task
        !$omp end task

        !$omp task depend(in: x) if(.FALSE.)    !!1st taskwait
        !$omp end task

        print*, "x=", x

        !$omp taskwait                          !!2nd taskwait

        print*, "y=", y

    end subroutine
end program
