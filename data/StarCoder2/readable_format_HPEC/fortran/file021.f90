program DRB028_privatemissing_orig_yes
    use omp_lib
    implicit none

    integer :: i, tmp, len
    integer, dimension(:), allocatable :: a

    len = 100
    allocate (a(len))

    do i = 1, len
        a(i) = i
    end do

    !$omp parallel do
    do i = 1, len
        tmp = a(i) + i
        a(i) = tmp
    end do
    !$omp end parallel do

    print 100, a(50)
    100 format ('a(50)=',i3)

    deallocate(a)
end program
