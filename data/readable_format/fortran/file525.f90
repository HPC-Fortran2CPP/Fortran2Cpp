PROGRAM testSaveKernelsOuterCore
    IMPLICIT NONE
    REAL(KIND=4) :: RHOSTORE_OUTER_CORE(5,5,5,1120)
    REAL(KIND=4) :: KAPPAVSTORE_OUTER_CORE(5,5,5,1120)
    REAL(KIND=4) :: RHO_KL_OUTER_CORE(5,5,5,1)
    REAL(KIND=4) :: ALPHA_KL_OUTER_CORE(5,5,5,1)

    ! Initialize some test values
    RHOSTORE_OUTER_CORE = 1.0
    KAPPAVSTORE_OUTER_CORE = 2.0
    RHO_KL_OUTER_CORE = 3.0
    ALPHA_KL_OUTER_CORE = 4.0
    
    ! Call the subroutine
    CALL SAVE_KERNELS_OUTER_CORE(RHOSTORE_OUTER_CORE, KAPPAVSTORE_OUTER_CORE, RHO_KL_OUTER_CORE, ALPHA_KL_OUTER_CORE)
    
    ! Output some results for manual verification
    PRINT *, 'RHO_KL_OUTER_CORE(1,1,1,1):', RHO_KL_OUTER_CORE(1,1,1,1)
    PRINT *, 'ALPHA_KL_OUTER_CORE(1,1,1,1):', ALPHA_KL_OUTER_CORE(1,1,1,1)

    CONTAINS

    SUBROUTINE SAVE_KERNELS_OUTER_CORE(RHOSTORE_OUTER_CORE, KAPPAVSTORE_OUTER_CORE, RHO_KL_OUTER_CORE, ALPHA_KL_OUTER_CORE)
        REAL(KIND=4), INTENT(INOUT) :: RHOSTORE_OUTER_CORE(5,5,5,1120)
        REAL(KIND=4), INTENT(INOUT) :: KAPPAVSTORE_OUTER_CORE(5,5,5,1120)
        REAL(KIND=4), INTENT(INOUT) :: RHO_KL_OUTER_CORE(5,5,5,1)
        REAL(KIND=4), INTENT(INOUT) :: ALPHA_KL_OUTER_CORE(5,5,5,1)
        ! Add your subroutine logic here.
    END SUBROUTINE SAVE_KERNELS_OUTER_CORE

END PROGRAM testSaveKernelsOuterCore