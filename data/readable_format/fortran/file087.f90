! tableb_mod.f90
MODULE tableb_mod
  INTERFACE
    SUBROUTINE TABLEB(X,Y,VER,SCALE,REFVAL,WIDTH,IFORMAT,NAME,UNIT)
      IMPLICIT NONE
      INTEGER         ,INTENT(IN)  :: X
      INTEGER         ,INTENT(IN)  :: Y
      INTEGER         ,INTENT(IN)  :: VER
      INTEGER         ,INTENT(OUT) :: SCALE
      INTEGER         ,INTENT(OUT) :: REFVAL
      INTEGER         ,INTENT(OUT) :: WIDTH
      CHARACTER(LEN=*),INTENT(OUT) :: IFORMAT
      CHARACTER(LEN=*),INTENT(OUT) :: NAME
      CHARACTER(LEN=*),INTENT(OUT) :: UNIT
    END SUBROUTINE TABLEB
  END INTERFACE
END MODULE tableb_mod

SUBROUTINE TABLEB(X,Y,VER,SCALE,REFVAL,WIDTH,IFORMAT,NAME,UNIT)
  IMPLICIT NONE
  INTEGER         ,INTENT(IN)  :: X
  INTEGER         ,INTENT(IN)  :: Y
  INTEGER         ,INTENT(IN)  :: VER
  INTEGER         ,INTENT(OUT) :: SCALE
  INTEGER         ,INTENT(OUT) :: REFVAL
  INTEGER         ,INTENT(OUT) :: WIDTH
  CHARACTER(LEN=*),INTENT(OUT) :: IFORMAT
  CHARACTER(LEN=*),INTENT(OUT) :: NAME
  CHARACTER(LEN=*),INTENT(OUT) :: UNIT
  
  SCALE = X + Y
  REFVAL = X - Y
  WIDTH = X * Y
  IFORMAT = 'Format-A'
  NAME = 'TestName'
  UNIT = 'Unit-X'
END SUBROUTINE TABLEB

! tableb_test.f90
PROGRAM test_tableb
  USE tableb_mod
  IMPLICIT NONE
  INTEGER :: X, Y, VER, SCALE, REFVAL, WIDTH
  CHARACTER(LEN=20) :: IFORMAT, NAME, UNIT
  
  X = 5
  Y = 3
  VER = 1
  
  CALL TABLEB(X, Y, VER, SCALE, REFVAL, WIDTH, IFORMAT, NAME, UNIT)
  
  PRINT *, "SCALE: ", SCALE
  PRINT *, "REFVAL: ", REFVAL
  PRINT *, "WIDTH: ", WIDTH
  PRINT *, "IFORMAT: ", IFORMAT
  PRINT *, "NAME: ", NAME
  PRINT *, "UNIT: ", UNIT
END PROGRAM test_tableb