program DRB001_antidep1_orig_yes
use omp_lib
    implicit none
    integer :: i, len
    integer :: a(1000)

    len = 1000

    do i = 1, len
        a(i) = i
    end do

    !$omp parallel do
    do i = 1, len-1
        a(i) = a(i+1) + 1
    end do
    !$omp end parallel do

    print 100, a(500)
    100 format ('a(500)=',i3)
end program
