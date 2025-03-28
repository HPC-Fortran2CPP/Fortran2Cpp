PROGRAM TestLBTORD
  IMPLICIT NONE
  DOUBLE PRECISION :: LRAD, BRAD, RA, DEC
  
  ! Test 1
  PRINT *, "Test 1"
  LRAD = 1.0
  BRAD = 0.5
  CALL LBTORD(LRAD, BRAD, RA, DEC)
  PRINT *, "LRAD:", LRAD, "BRAD:", BRAD, "RA:", RA, "DEC:", DEC
  
  ! Test 2
  PRINT *, "Test 2"
  LRAD = 2.0
  BRAD = 1.0
  CALL LBTORD(LRAD, BRAD, RA, DEC)
  PRINT *, "LRAD:", LRAD, "BRAD:", BRAD, "RA:", RA, "DEC:", DEC
  
  ! Add more tests as needed with different LRAD, BRAD values
  
CONTAINS

  SUBROUTINE LBTORD(LRAD, BRAD, RA, DEC)
    IMPLICIT NONE
    DOUBLE PRECISION, INTENT(IN) :: LRAD, BRAD
    DOUBLE PRECISION, INTENT(OUT) :: RA, DEC
    DOUBLE PRECISION :: X, Y, SINA
    DOUBLE PRECISION :: PI, TWOPI
    DOUBLE PRECISION :: CON27, CON33, CON192
    
    ! Constants
    PI = 3.1415926535898D0
    TWOPI = 2.D0*PI
    CON27 = 27.40D0*PI/180.D0
    CON33 = 33.00D0*PI/180.D0
    CON192 = 192.25D0*PI/180.D0
    
    ! Computation
    SINA = DSIN(BRAD)*DSIN(CON27) + DCOS(BRAD)*DCOS(CON27)*DSIN(LRAD-CON33)
    X = DCOS(BRAD)*DCOS(LRAD-CON33)
    Y = DSIN(BRAD)*DCOS(CON27) - DCOS(BRAD)*DSIN(CON27)*DSIN(LRAD-CON33)
    RA = DATAN2(X,Y)
    RA = RA + CON192
    DEC = DASIN(SINA)
    RA = MOD(RA+TWOPI,TWOPI)
  END SUBROUTINE LBTORD

END PROGRAM TestLBTORD