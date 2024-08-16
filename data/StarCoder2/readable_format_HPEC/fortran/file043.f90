program DRB163_simdmissinglock1_orig_gpu_no
    use omp_lib
    use DRB163
    implicit none

    do i = 1, 16
        var(i) = 0
    end do

    !$omp target map(tofrom:var) device(0)
    !$omp teams distribute parallel do reduction(+:var)
    do i = 1, 20
        !$omp simd
        do j = 1, 16
            var(j) = var(j)+1
        end do
        !$omp end simd
    end do
    !$omp end teams distribute parallel do
    !$omp end target

    do i = 1, 16
        if (var(i) /= 20) then
            print*, var(i), i
        end if
    end do

end program
