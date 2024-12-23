! Define the SDAWTS subroutine
SUBROUTINE SDAWTS(NEQ, IWT, RTOL, ATOL, Y, WT, RPAR, IPAR)
  INTEGER NEQ, IWT, IPAR(*)
  REAL RTOL(*), ATOL(*), Y(*), WT(*), RPAR(*)
  INTEGER I
  REAL ATOLI, RTOLI
  RTOLI = RTOL(1)
  ATOLI = ATOL(1)
  DO I = 1, NEQ
     IF (IWT .EQ. 0) THEN
       WT(I) = RTOLI*ABS(Y(I)) + ATOLI
     ELSE
       RTOLI = RTOL(I)
       ATOLI = ATOL(I)
       WT(I) = RTOLI*ABS(Y(I)) + ATOLI
     END IF
  END DO
  RETURN
END SUBROUTINE SDAWTS

! Main program to test SDAWTS
PROGRAM TestSDAWTS
  IMPLICIT NONE
  INTEGER :: NEQ, IWT
  REAL, DIMENSION(:), ALLOCATABLE :: RTOL, ATOL, Y, WT
  INTEGER, DIMENSION(:), ALLOCATABLE :: IPAR
  REAL, DIMENSION(:), ALLOCATABLE :: RPAR
  INTEGER :: I

  ! Initialize test case
  NEQ = 3
  IWT = 0
  ALLOCATE(RTOL(NEQ), ATOL(NEQ), Y(NEQ), WT(NEQ), RPAR(1), IPAR(1))
  RTOL = (/0.1, 0.1, 0.1/)
  ATOL = (/1.0, 1.0, 1.0/)
  Y = (/10.0, 20.0, 30.0/)

  ! Call the SDAWTS subroutine
  CALL SDAWTS(NEQ, IWT, RTOL, ATOL, Y, WT, RPAR, IPAR)

  ! Output the results
  PRINT *, "WT for test case:"
  DO I = 1, NEQ
     PRINT *, WT(I)
  END DO

  ! Clean up
  DEALLOCATE(RTOL, ATOL, Y, WT, RPAR, IPAR)
END PROGRAM TestSDAWTS