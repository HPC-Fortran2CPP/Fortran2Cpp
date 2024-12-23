module e_50_3_mod
contains
  subroutine init(v1, v2, N)
    implicit none
    integer :: i, N
    real :: v1(N), v2(N)
    do i = 1, N
      v1(i) = i + 2.0
      v2(i) = i - 3.0
    end do
  end subroutine

  subroutine check(p, N)
    implicit none
    integer :: i, N
    real, parameter :: EPS = 0.00001
    real :: diff, p(N)
    do i = 1, N
      diff = p(i) - (i + 2.0) * (i - 3.0)
      if (diff > EPS .or. -diff > EPS) then
        print *, "Check failed at index ", i
        call abort()
      endif
    end do
  end subroutine

  subroutine vec_mult(N)
    implicit none
    integer :: i, N
    real :: p(N), v1(N), v2(N)
    call init(v1, v2, N)
    
    ! Parallelize this loop without targeting GPU offloading
    !$omp parallel do
    do i = 1, N
      p(i) = v1(i) * v2(i)
    end do
    !$omp end parallel do

    call check(p, N)
  end subroutine
end module

program e_50_3
  use e_50_3_mod, only : vec_mult
  implicit none
  integer :: n
  n = 1000
  call vec_mult(n)
  print *, "All tests passed!"
end program