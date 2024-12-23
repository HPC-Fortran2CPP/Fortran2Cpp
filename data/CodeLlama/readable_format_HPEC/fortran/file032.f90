program DRB069_sectionslock1_orig_no
    use omp_lib
    implicit none

    integer (kind=omp_lock_kind) lock
    integer :: i
    i = 0
    call omp_init_lock(lock)

    !$omp parallel sections
        !$omp section
        call omp_set_lock(lock)
        i = i + 1
        call omp_unset_lock(lock)
        !$omp section
        call omp_set_lock(lock)
        i = i + 2
        call omp_unset_lock(lock)
    !$omp end parallel sections

    call omp_destroy_lock(lock)

    print 100, i
    100 format ('I =',i3)
end program
