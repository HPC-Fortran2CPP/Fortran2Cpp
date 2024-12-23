program DRB021_reductionmissing_orig_yes
    use omp_lib
    implicit none

    integer :: i, j, len
    real :: temp, getSum
    real, dimension (:,:), allocatable :: u

    len = 100
    getSum = 0.0

    allocate (u(len, len))

    do i = 1, len
        do j = 1, len
            u(i,j) = 0.5
        end do
    end do

    !$omp parallel do private(temp, i, j)
    do i = 1, len
        do j = 1, len
            temp = u(i,j)
            getSum = getSum + temp * temp
        end do
    end do
    !$omp end parallel do

    print*,"sum =", getSum
    deallocate(u)
end program
