PROGRAM TestDLAE2
  IMPLICIT NONE
  DOUBLE PRECISION A, B, C, RT1, RT2

  ! Test case 1
  A = 4.0D0
  B = 2.0D0
  C = 1.0D0
  CALL DLAE2(A, B, C, RT1, RT2)
  PRINT *, "Test Case 1: A=", A, " B=", B, " C=", C
  PRINT *, "RT1=", RT1, " RT2=", RT2

  ! Add more test cases as needed

END PROGRAM TestDLAE2


SUBROUTINE DLAE2( A, B, C, RT1, RT2 )
  DOUBLE PRECISION A, B, C, RT1, RT2
  DOUBLE PRECISION ONE
  PARAMETER (ONE = 1.0D0)
  DOUBLE PRECISION TWO
  PARAMETER (TWO = 2.0D0)
  DOUBLE PRECISION ZERO
  PARAMETER (ZERO = 0.0D0)
  DOUBLE PRECISION HALF
  PARAMETER (HALF = 0.5D0)
  DOUBLE PRECISION AB, ACMN, ACMX, ADF, DF, RT, SM, TB
  INTRINSIC ABS, SQRT

  SM = A + C
  DF = A - C
  ADF = ABS(DF)
  TB = B + B
  AB = ABS(TB)
  IF (ABS(A) > ABS(C)) THEN
     ACMX = A
     ACMN = C
  ELSE
     ACMX = C
     ACMN = A
  END IF
  IF (ADF > AB) THEN
     RT = ADF*SQRT(ONE + (AB / ADF)**2)
  ELSE IF (ADF < AB) THEN
     RT = AB*SQRT(ONE + (ADF / AB)**2)
  ELSE
     RT = AB*SQRT(TWO)
  END IF
  IF (SM < ZERO) THEN
     RT1 = HALF*(SM - RT)
     RT2 = (ACMX / RT1)*ACMN - (B / RT1)*B
  ELSE IF (SM > ZERO) THEN
     RT1 = HALF*(SM + RT)
     RT2 = (ACMX / RT1)*ACMN - (B / RT1)*B
  ELSE
     RT1 = HALF*RT
     RT2 = -HALF*RT
  END IF
  RETURN
END SUBROUTINE DLAE2