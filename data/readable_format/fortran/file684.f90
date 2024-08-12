DOUBLE PRECISION FUNCTION PROBKS(ALAM)
      DOUBLE PRECISION ALAM
      PARAMETER (EPS1=0.001, EPS2=1.E-8)
      DOUBLE PRECISION A2, FAC, TERM, TERMBF
      INTEGER J
      A2=-2.*ALAM**2
      FAC=2.
      PROBKS=0.
      TERMBF=0.
      DO 11 J=1,100
        TERM=FAC*EXP(A2*J**2)
        PROBKS=PROBKS+TERM
        IF(ABS(TERM).LT.EPS1*TERMBF.OR.ABS(TERM).LT.EPS2*PROBKS)RETURN
        FAC=-FAC
        TERMBF=ABS(TERM)
11    CONTINUE
      PROBKS=1.
      RETURN
      END

      PROGRAM TESTPROBKS
      IMPLICIT NONE
      DOUBLE PRECISION PROBKS
      DOUBLE PRECISION RESULT
      DOUBLE PRECISION ALAM

      ALAM = 0.5
      RESULT = PROBKS(ALAM)
      PRINT *, 'Test 1, ALAM = 0.5, Result = ', RESULT

      ALAM = 1.0
      RESULT = PROBKS(ALAM)
      PRINT *, 'Test 2, ALAM = 1.0, Result = ', RESULT

      END PROGRAM TESTPROBKS