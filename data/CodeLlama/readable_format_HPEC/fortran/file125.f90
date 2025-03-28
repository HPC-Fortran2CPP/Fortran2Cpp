program DRB054_inneronly2_orig_no
    use omp_lib
    implicit none

    integer i,j,n,m
    real, dimension(:,:), allocatable :: b

    n = 100
    m = 100

    allocate (b(n,m))

    do i = 1, n
        do j = 1, m
            b(i,j) = i*j
        end do
    end do

    do i = 2, n
        !$omp parallel do
        do j =2, m
            b(i,j)=b(i-1,j-1)
        end do
        !$omp end parallel do
    end do

    deallocate(b)
end program
