program DRB130_mergeable_taskwait_orig_no
    use omp_lib
    implicit none

    integer :: x
    x = 2

    !$omp task shared(x) mergeable
    x = x+1
    !$omp end task

    print 100, x
    100 format ('x =',3i8)
end program
