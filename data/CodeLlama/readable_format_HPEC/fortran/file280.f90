program DRB098_simd2_orig_no
    use omp_lib
    implicit none

    integer, parameter :: dp = kind(1.0d0)
    real(dp), dimension(:,:), allocatable :: a,b,c
    integer :: len, i, j

    len = 100
    allocate (a(len,len))
    allocate (b(len,len))
    allocate (c(len,len))

    do i = 1, len
        do j = 1, len
            a(i,j) = real(i,dp)/2.0
            b(i,j) = real(i,dp)/3.0
            c(i,j) = real(i,dp)/7.0
        end do
    end do

    !$omp simd collapse(2)
    do i = 1, len
        do j = 1, len
            c(i,j)=a(i,j)*b(i,j)
        end do
    end do
    !$omp end simd

    print*,'c(50,50) =',c(50,50)

    deallocate(a,b,c)
end program
