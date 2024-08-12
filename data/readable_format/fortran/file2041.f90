MODULE SETUP_EXACT_UNDOING_TO_DISK__genmod
  LOGICAL :: setupDone = .FALSE.
  
  CONTAINS
  
  SUBROUTINE SETUP_EXACT_UNDOING_TO_DISK()
    setupDone = .TRUE.
  END SUBROUTINE SETUP_EXACT_UNDOING_TO_DISK

END MODULE SETUP_EXACT_UNDOING_TO_DISK__genmod

PROGRAM testSetup
  USE SETUP_EXACT_UNDOING_TO_DISK__genmod
  
  CALL SETUP_EXACT_UNDOING_TO_DISK()
  
  IF (setupDone) THEN
    PRINT *, "Test Passed: setupDone is TRUE."
  ELSE
    PRINT *, "Test Failed: setupDone is FALSE."
  END IF
  
END PROGRAM testSetup