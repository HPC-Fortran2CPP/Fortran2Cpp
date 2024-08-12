MODULE MathOperations
  IMPLICIT NONE
CONTAINS
  INTEGER FUNCTION add(a, b)
    INTEGER, INTENT(IN) :: a, b
    add = a + b
  END FUNCTION add
END MODULE MathOperations

PROGRAM MAIN
  USE MathOperations
  IMPLICIT NONE
  PRINT *, add(2, 3)
END PROGRAM MAIN