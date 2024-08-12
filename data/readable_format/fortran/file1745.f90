MODULE nrtype
  IMPLICIT NONE
  SAVE
  INTEGER, PARAMETER :: I4B = SELECTED_INT_KIND(9)
  INTEGER, PARAMETER :: I2B = SELECTED_INT_KIND(4)
  INTEGER, PARAMETER :: I1B = SELECTED_INT_KIND(2)
  INTEGER, PARAMETER :: SP = KIND(1.0)
  INTEGER, PARAMETER :: DP = KIND(1.0D0)
  INTEGER, PARAMETER :: QP = SELECTED_REAL_KIND(32)
  INTEGER, PARAMETER :: SPC = KIND((1.0,1.0))
  INTEGER, PARAMETER :: DPC = KIND((1.0D0,1.0D0))
  INTEGER, PARAMETER :: LGT = KIND(.true.)
  REAL(SP), PARAMETER :: PI=3.141592653589793238462643383279502884197_sp
  REAL(SP), PARAMETER :: PIO2=1.57079632679489661923132169163975144209858_sp
  REAL(SP), PARAMETER :: TWOPI=6.283185307179586476925286766559005768394_sp
  REAL(SP), PARAMETER :: SQRT2=1.41421356237309504880168872420969807856967_sp
  REAL(SP), PARAMETER :: EULER=0.5772156649015328606065120900824024310422_sp
  REAL(DP), PARAMETER :: PI_D=3.141592653589793238462643383279502884197_dp
  REAL(DP), PARAMETER :: PIO2_D=1.57079632679489661923132169163975144209858_dp
  REAL(DP), PARAMETER :: TWOPI_D=6.283185307179586476925286766559005768394_dp
END MODULE nrtype

PROGRAM test_nrtype
  USE nrtype
  IMPLICIT NONE

  PRINT *, 'Testing nrtype module...'
  PRINT *, 'Sizes (bytes) and ranges of integer types:'
  PRINT *, 'I4B:', SIZEOF(I4B), ', Range: ', HUGE(0_I4B)
  PRINT *, 'I2B:', SIZEOF(I2B), ', Range: ', HUGE(0_I2B)
  PRINT *, 'I1B:', SIZEOF(I1B), ', Range: ', HUGE(0_I1B)
  
  PRINT *, 'Precision of floating point types (bytes):'
  PRINT *, 'SP:', SIZEOF(0.0_SP)
  PRINT *, 'DP:', SIZEOF(0.0_DP)
  PRINT *, 'QP:', SIZEOF(0.0_QP)

  PRINT *, 'Mathematical constants (SP):'
  PRINT *, 'PI = ', PI
  PRINT *, 'PIO2 = ', PIO2
  PRINT *, 'TWOPI = ', TWOPI
  PRINT *, 'SQRT2 = ', SQRT2
  PRINT *, 'EULER = ', EULER

  PRINT *, 'Mathematical constants (DP):'
  PRINT *, 'PI_D = ', PI_D
  PRINT *, 'PIO2_D = ', PIO2_D
  PRINT *, 'TWOPI_D = ', TWOPI_D
END PROGRAM test_nrtype