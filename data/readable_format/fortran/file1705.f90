PROGRAM test_scnrm2
    COMPLEX, ALLOCATABLE :: X(:)
    INTEGER :: N, INCX
    REAL :: RESULT

    ! Test 1
    N = 3
    INCX = 1
    ALLOCATE(X(N))
    X = [(COMPLEX(i, i), i=1,N)]
    RESULT = SCNRM2(N, X, INCX)
    PRINT *, "Test 1 Result: ", RESULT
    DEALLOCATE(X)

    ! Test 2
    N = 4
    INCX = 2
    ALLOCATE(X(N))
    X = [(COMPLEX(i*0.5, -i*0.5), i=1,N)]
    RESULT = SCNRM2(N, X, INCX)
    PRINT *, "Test 2 Result: ", RESULT
    DEALLOCATE(X)

CONTAINS

    REAL FUNCTION SCNRM2( N, X, INCX )
        INTEGER INCX, N
        COMPLEX X( * )
        REAL ONE, ZERO
        PARAMETER ( ONE = 1.0E+0, ZERO = 0.0E+0 )
        INTEGER IX
        REAL NORM, SCALE, SSQ, TEMP
        INTRINSIC ABS, AIMAG, REAL, SQRT
        IF( N.LT.1 .OR. INCX.LT.1 )THEN
            NORM  = ZERO
        ELSE
            SCALE = ZERO
            SSQ   = ONE
            DO 10, IX = 1, 1 + ( N - 1 )*INCX, INCX
                IF( REAL( X( IX ) ).NE.ZERO )THEN
                    TEMP = ABS( REAL( X( IX ) ) )
                    IF( SCALE.LT.TEMP )THEN
                        SSQ   = ONE   + SSQ*( SCALE/TEMP )**2
                        SCALE = TEMP
                    ELSE
                        SSQ   = SSQ   +     ( TEMP/SCALE )**2
                    END IF
                END IF
                IF( AIMAG( X( IX ) ).NE.ZERO )THEN
                    TEMP = ABS( AIMAG( X( IX ) ) )
                    IF( SCALE.LT.TEMP )THEN
                        SSQ   = ONE   + SSQ*( SCALE/TEMP )**2
                        SCALE = TEMP
                    ELSE
                        SSQ   = SSQ   +     ( TEMP/SCALE )**2
                    END IF
                END IF
    10      CONTINUE
            NORM  = SCALE * SQRT( SSQ )
        END IF
        SCNRM2 = NORM
        RETURN
    END FUNCTION SCNRM2
END PROGRAM test_scnrm2