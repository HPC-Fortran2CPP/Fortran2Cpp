MODULE SET_THETAPHI_RANGE__genmod
  IMPLICIT NONE
CONTAINS
  SUBROUTINE SET_THETAPHI_RANGE(THETA, PHI, USE_POSITIVE_LON)
    REAL(KIND=8), INTENT(INOUT) :: THETA
    REAL(KIND=8), INTENT(INOUT) :: PHI
    LOGICAL, INTENT(IN) :: USE_POSITIVE_LON
    
    ! Assuming the adjustment logic as described before
    IF (USE_POSITIVE_LON .AND. PHI < 0.0D0) THEN
      PHI = PHI + 360.0D0
    ELSE IF (.NOT. USE_POSITIVE_LON .AND. PHI > 0.0D0) THEN
      PHI = PHI - 360.0D0
    END IF
  END SUBROUTINE SET_THETAPHI_RANGE
END MODULE SET_THETAPHI_RANGE__genmod

PROGRAM test_SET_THETAPHI_RANGE
  USE SET_THETAPHI_RANGE__genmod
  IMPLICIT NONE
  REAL(KIND=8) :: THETA, PHI
  
  ! Test case 1: Negative PHI, USE_POSITIVE_LON = .TRUE.
  THETA = 0.0
  PHI = -100.0
  CALL SET_THETAPHI_RANGE(THETA, PHI, .TRUE.)
  PRINT *, "Test 1: THETA=", THETA, " PHI=", PHI  ! Expected PHI: 260.0
  
  ! Test case 2: Positive PHI, USE_POSITIVE_LON = .FALSE.
  THETA = 0.0
  PHI = 100.0
  CALL SET_THETAPHI_RANGE(THETA, PHI, .FALSE.)
  PRINT *, "Test 2: THETA=", THETA, " PHI=", PHI  ! Expected PHI: -260.0
  
  ! Add more tests as needed
END PROGRAM test_SET_THETAPHI_RANGE