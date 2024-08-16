program DRB146_atomicupdate_orig_gpu_no
    use omp_lib
    implicit none

    integer :: var, i
    var = 0

    !$omp target map(tofrom:var) device(0)
        !$omp teams distribute
        do i = 1, 100
            !$omp atomic update
            var = var+1
            !$omp end atomic
        end do
        !$omp end teams distribute
    !$omp end target

  print*,var
end program
