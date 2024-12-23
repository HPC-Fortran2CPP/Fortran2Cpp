program DRB010_lastprivatemissing_var_yes
    use omp_lib
    implicit none

    integer :: i, len, argCount, allocStatus, rdErr, x, ix
    character(len=80), dimension(:), allocatable :: args
    len = 10000

    argCount = command_argument_count()
    if (argCount == 0) then
        write (*,'(a)') "No command line arguments provided."
    end if

    allocate(args(argCount), stat=allocStatus)
    if (allocStatus > 0) then
        write (*,'(a)') "Allocation error, program terminated."
        stop
    end if

    do ix = 1, argCount
        call get_command_argument(ix,args(ix))
    end do

    if (argCount >= 1) then
        read (args(1), '(i10)', iostat=rdErr) len
        if (rdErr /= 0 ) then
            write (*,'(a)') "Error, invalid integer value."
        end if
    end if

    !$omp parallel do private(i)
    do i = 0, len
        x = i
    end do
    !$omp end parallel do
    write(*,*) 'x =', x

    deallocate(args)
end program
