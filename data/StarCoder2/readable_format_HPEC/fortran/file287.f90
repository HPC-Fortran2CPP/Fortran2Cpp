subroutine print_array(m, symmat)
implicit none

DATA_TYPE, dimension(N, M) :: symmat
integer :: m
integer :: i, j
do i = 1, m
do j = 1, m
write(0, DATA_PRINTF_MODIFIER) symmat(j, i)
if (mod(((i - 1) * m) + j - 1, 20) == 0) then
write(0, *)
end if
end do
end do
write(0, *)
end subroutine
