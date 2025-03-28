PROGRAM test_iau_PR00
  IMPLICIT NONE
  
  DOUBLE PRECISION DATE1, DATE2, DPSIPR, DEPSPR
  
  ! Test case
  DATE1 = 2451545.D0
  DATE2 = 0.D0
  
  CALL iau_PR00(DATE1, DATE2, DPSIPR, DEPSPR)
  
  PRINT *, 'DPSIPR = ', DPSIPR
  PRINT *, 'DEPSPR = ', DEPSPR

END PROGRAM test_iau_PR00

SUBROUTINE iau_PR00 ( DATE1, DATE2, DPSIPR, DEPSPR )

  IMPLICIT NONE

  DOUBLE PRECISION DATE1, DATE2, DPSIPR, DEPSPR
  DOUBLE PRECISION DAS2R, DJ0, DJC, T, PRECOR, OBLCOR

  ! Constants
  DAS2R = 4.848136811095359935899141D-6
  DJ0 = 2451545D0
  DJC = 36525D0
  PRECOR = -0.29965D0 * DAS2R
  OBLCOR = -0.02524D0 * DAS2R

  T = ( ( DATE1-DJ0 ) + DATE2 ) / DJC

  DPSIPR = PRECOR * T
  DEPSPR = OBLCOR * T

END SUBROUTINE iau_PR00