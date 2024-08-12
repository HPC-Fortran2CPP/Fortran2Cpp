MODULE UPDATE_VELOC_ACOUSTIC_LDDRK_BACKWARD__genmod
  IMPLICIT NONE
CONTAINS
  SUBROUTINE UPDATE_VELOC_ACOUSTIC_LDDRK_BACKWARD(result)
    INTEGER, INTENT(OUT) :: result
    result = 100
  END SUBROUTINE UPDATE_VELOC_ACOUSTIC_LDDRK_BACKWARD
END MODULE UPDATE_VELOC_ACOUSTIC_LDDRK_BACKWARD__genmod

PROGRAM test
  USE UPDATE_VELOC_ACOUSTIC_LDDRK_BACKWARD__genmod
  INTEGER :: result_value
  
  CALL UPDATE_VELOC_ACOUSTIC_LDDRK_BACKWARD(result_value)
  
  IF (result_value == 100) THEN
    PRINT *, "Test passed with result_value =", result_value
  ELSE
    PRINT *, "Test failed. result_value =", result_value
  END IF
END PROGRAM test