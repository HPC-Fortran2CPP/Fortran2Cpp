PROGRAM TestTEMPDATA
    IMPLICIT NONE

    INTEGER, PARAMETER :: KS = 10
    REAL :: HP
    REAL, DIMENSION(KS) :: TT2A, PT2A, WG2A, ETAS, ETAR
    REAL :: HP1, TT123(48), PT123(48), WG123(48), ETAS123(48), ETAR123(48)
    INTEGER :: KS1
    INTEGER :: I

    COMMON /DATTRF/ HP1, TT123, PT123, WG123, ETAS123, ETAR123, KS1

    ! Initialize input data
    HP = 1.23
    DO I = 1, KS
        TT2A(I) = REAL(I)
        PT2A(I) = REAL(I) + 0.1
        WG2A(I) = REAL(I) + 0.2
        ETAS(I) = REAL(I) + 0.3
        ETAR(I) = REAL(I) + 0.4
    END DO

    ! Call the subroutine
    CALL TEMPDATA(HP, TT2A, PT2A, WG2A, ETAS, ETAR, KS)

    ! Verify results
    PRINT *, 'HP1: ', HP1
    PRINT *, 'KS1: ', KS1
    PRINT *, 'ETAS123: ', ETAS123(1:KS)
    PRINT *, 'ETAR123: ', ETAR123(1:KS)

END PROGRAM TestTEMPDATA

SUBROUTINE TEMPDATA(HP, TT2A, PT2A, WG2A, ETAS, ETAR, KS)
    IMPLICIT NONE

    INTEGER, INTENT(IN) :: KS
    REAL, INTENT(IN) :: HP
    REAL, DIMENSION(KS), INTENT(IN) :: TT2A, PT2A, WG2A, ETAS, ETAR
    REAL :: HP1, TT123(48), PT123(48), WG123(48), ETAS123(48), ETAR123(48)
    INTEGER :: KS1
    INTEGER :: I

    COMMON /DATTRF/ HP1, TT123, PT123, WG123, ETAS123, ETAR123, KS1

    KS1 = KS
    HP1 = HP

    DO I = 1, KS
        ETAS123(I) = ETAS(I)
        ETAR123(I) = ETAR(I)
    END DO
END SUBROUTINE TEMPDATA