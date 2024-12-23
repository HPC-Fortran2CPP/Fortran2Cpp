MODULE CHEMMF_MOD
  IMPLICIT NONE
CONTAINS
  SUBROUTINE CHEMMF(SIDE, UPLO, M, N, ALPHA, A, LDA, B, LDB, BETA, C, LDC)
    CHARACTER(LEN=1), INTENT(IN) :: SIDE, UPLO
    INTEGER, INTENT(IN) :: M, N, LDA, LDB, LDC
    COMPLEX, INTENT(IN) :: ALPHA, BETA
    COMPLEX, INTENT(IN) :: A(LDA, *), B(LDB, *)
    COMPLEX, INTENT(INOUT) :: C(LDC, *)
    INTEGER :: i, j

    IF (SIDE == 'L') THEN
      DO i = 1, M
        DO j = 1, N
          C(i, j) = ALPHA * A(i, j) + BETA * C(i, j)
        END DO
      END DO
    ELSE
      PRINT *, "SIDE not implemented."
    END IF
  END SUBROUTINE CHEMMF
END MODULE CHEMMF_MOD

PROGRAM TEST_CHEMMF
  USE CHEMMF_MOD
  IMPLICIT NONE
  CHARACTER(LEN=1) :: SIDE, UPLO
  INTEGER :: M, N, LDA, LDB, LDC
  COMPLEX :: ALPHA, BETA
  COMPLEX, ALLOCATABLE :: A(:,:), B(:,:), C(:,:)

  SIDE = 'L'
  UPLO = 'U'
  M = 2
  N = 2
  LDA = 2
  LDB = 2
  LDC = 2
  ALPHA = (1.0, 0.0)
  BETA = (0.5, 0.0)

  ALLOCATE(A(LDA,N), B(LDB,N), C(LDC,N))
  A = RESHAPE([ (1.0,0.0), (2.0,0.0), (3.0,0.0), (4.0,0.0) ], SHAPE(A))
  B = RESHAPE([ (5.0,0.0), (6.0,0.0), (7.0,0.0), (8.0,0.0) ], SHAPE(B))
  C = RESHAPE([ (9.0,0.0), (10.0,0.0), (11.0,0.0), (12.0,0.0) ], SHAPE(C))

  CALL CHEMMF(SIDE, UPLO, M, N, ALPHA, A, LDA, B, LDB, BETA, C, LDC)

  PRINT *, 'Matrix C after CHEMMF:'
  PRINT *, C

  DEALLOCATE(A, B, C)
END PROGRAM TEST_CHEMMF