PROGRAM test_iau_PR00
  IMPLICIT NONE
  DOUBLE PRECISION DATE1, DATE2, DPSIPR, DEPSPR

  ! Test case
  DATE1 = 2451545.0
  DATE2 = 0.0

  CALL iau_PR00(DATE1, DATE2, DPSIPR, DEPSPR)

  PRINT *, 'DPSIPR = ', DPSIPR
  PRINT *, 'DEPSPR = ', DEPSPR

  ! Here you can add assertions or checks if available in your Fortran environment
  ! For simple visual inspection, ensure the printed values meet expected outcomes based on manual calculations or known results.

END PROGRAM test_iau_PR00


SUBROUTINE iau_PR00 ( DATE1, DATE2, DPSIPR, DEPSPR )
  IMPLICIT NONE
  DOUBLE PRECISION DATE1, DATE2, DPSIPR, DEPSPR

  DOUBLE PRECISION DAS2R
  PARAMETER (DAS2R = 4.848136811095359935899141D-6)

  DOUBLE PRECISION DJ00
  PARAMETER (DJ00 = 2451545D0)

  DOUBLE PRECISION DJC
  PARAMETER (DJC = 36525D0)

  DOUBLE PRECISION T

  DOUBLE PRECISION PRECOR, OBLCOR
  PARAMETER (PRECOR = -0.29965D0 * DAS2R, OBLCOR = -0.02524D0 * DAS2R)

  T = ((DATE1 - DJ00) + DATE2) / DJC

  DPSIPR = PRECOR * T
  DEPSPR = OBLCOR * T

END SUBROUTINE iau_PR00