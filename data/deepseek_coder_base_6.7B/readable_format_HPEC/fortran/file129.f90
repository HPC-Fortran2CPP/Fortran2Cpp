program DRB107_taskgroup_orig_no
    use omp_lib
    implicit none

    integer result
    result = 0

    !$omp parallel
    !$omp single
        !$omp taskgroup
            !$omp task
            call sleep(3)
            result = 1
            !$omp end task
        !$omp end taskgroup
        !$omp task
        result = 2
        !$omp end task
    !$omp end single
    !$omp end parallel

    print 100, result
    100 format ('result =',3i8)

end program
