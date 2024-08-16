program DRB159_nobarrier_orig_gpu_no
    use omp_lib
    use DRB159
    implicit none

    do i = 1, 8
        b(i) = 0
        c(i) = 2
        temp(i) = 0
    end do

    a = 2

    !$omp target map(tofrom:b) map(to:c,temp,a) device(0)
    !$omp parallel
    do i = 1, 100
        !$omp do
        do j = 1, 8
            temp(j) = b(j)+c(j)
        end do
        !$omp end do

        !$omp do
        do j = 8, 1, k-1
            b(j) = temp(j)*a
        end do
        !$omp end do
    end do
    !$omp end parallel
    !$omp end target

    do i = 1, 100
        val = val+2
        val = val*2
    end do

    do i = 1, 8
        if (val /= b(i)) then
            print*,b(i),val
        end if
    end do

end program
