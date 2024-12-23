PROGRAM testSLARRA
  IMPLICIT NONE
  INTEGER, PARAMETER :: N = 5
  REAL, DIMENSION(N) :: D = (/1.0, 2.0, 3.0, 4.0, 5.0/)
  REAL, DIMENSION(N-1) :: E = (/0.1, 0.2, 0.3, 0.4/)
  REAL, DIMENSION(N-1) :: E2
  REAL :: SPLTOL = 0.15
  REAL :: TNRM = 5.0
  INTEGER :: NSPLIT, INFO, I
  INTEGER, DIMENSION(N) :: ISPLIT

  DO I = 1, N-1
     E2(I) = E(I)**2
  END DO

  CALL SLARRA(N, D, E, E2, SPLTOL, TNRM, NSPLIT, ISPLIT, INFO)

  PRINT *, "INFO:", INFO
  PRINT *, "NSPLIT:", NSPLIT
  PRINT *, "ISPLIT:", ISPLIT
END PROGRAM testSLARRA

SUBROUTINE SLARRA(N, D, E, E2, SPLTOL, TNRM, NSPLIT, ISPLIT, INFO)
  INTEGER, INTENT(IN) :: N
  REAL, INTENT(IN) :: D(*), SPLTOL, TNRM
  REAL, INTENT(INOUT) :: E(*), E2(*) ! Changed from INTENT(IN) to INTENT(INOUT)
  INTEGER, INTENT(OUT) :: NSPLIT, ISPLIT(*), INFO
  REAL :: ZERO = 0.0E0
  INTEGER :: I
  REAL :: EABS, TMP1

  INFO = 0

  IF (N.LE.0) THEN
     RETURN
  END IF

  NSPLIT = 1
  IF (SPLTOL.LT.ZERO) THEN
     TMP1 = ABS(SPLTOL) * TNRM
     DO I = 1, N-1
        EABS = ABS(E(I))
        IF (EABS.LE.TMP1) THEN
           E(I) = ZERO
           E2(I) = ZERO
           ISPLIT(NSPLIT) = I
           NSPLIT = NSPLIT + 1
        END IF
     END DO
  ELSE
     DO I = 1, N-1
        EABS = ABS(E(I))
        IF (EABS.LE.SPLTOL * SQRT(ABS(D(I))) * SQRT(ABS(D(I+1)))) THEN
           E(I) = ZERO
           E2(I) = ZERO
           ISPLIT(NSPLIT) = I
           NSPLIT = NSPLIT + 1
        END IF
     END DO
  END IF
  ISPLIT(NSPLIT) = N
END SUBROUTINE SLARRA