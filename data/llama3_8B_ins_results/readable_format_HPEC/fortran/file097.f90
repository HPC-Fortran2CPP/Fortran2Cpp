program DRB147_critical1_orig_gpu_no
    use omp_lib
    implicit none

    integer :: var, i
    var = 0

    !$omp target map(tofrom:var) device(0)
    !$omp teams distribute parallel do
        do i = 0, 100
            !$omp atomic
            var = var+1
            !$omp atomic
            var = var-2
        end do
    !$omp end teams distribute parallel do
    !$omp end target

    print*, var
end program