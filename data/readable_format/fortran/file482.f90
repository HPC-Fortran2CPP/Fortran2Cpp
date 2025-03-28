MODULE PARS
  INTEGER, PARAMETER :: DP=8
  INTEGER, PARAMETER :: NMAX=50
  INTEGER, PARAMETER :: LMAX=0
  INTEGER, PARAMETER :: GMAX=150
  INTEGER :: NUSE, LUSE, GUSEX, GUSEY
  REAL(KIND=DP) :: BOSC, HBAR2M, HBROMG
  REAL(KIND=DP) :: KAPPAR, KAPPAS, V0R, V0S
  CHARACTER(LEN=100) :: QUAD
END MODULE PARS

MODULE GAUSS
  USE PARS
  REAL(KIND=DP) :: WX(1:GMAX), WY(1:GMAX)
  REAL(KIND=DP) :: XMESH(1:GMAX), YMESH(1:GMAX)
END MODULE GAUSS

MODULE LOGS
  USE PARS
  REAL(KIND=DP) :: LOGANL(0:NMAX,0:LMAX)
END MODULE LOGS

MODULE LX1D
  USE PARS
  REAL(KIND=DP) :: X1D(1:GMAX)
  REAL(KIND=DP) :: L1D(0:NMAX,0:LMAX,1:GMAX)
END MODULE LX1D

MODULE HRHOMAT
  USE PARS
  REAL(KIND=DP) :: HMAT(0:NMAX,0:NMAX)
  REAL(KIND=DP) :: RHOS(0:NMAX,0:NMAX)
END MODULE HRHOMAT

MODULE TVUMAT
  USE PARS
  REAL(KIND=DP) :: TUMAT(0:NMAX,0:NMAX)
  REAL(KIND=DP) :: VMAT(0:NMAX,0:NMAX,0:NMAX,0:NMAX)
  REAL(KIND=DP) :: V1MAT(0:NMAX,0:NMAX)
END MODULE TVUMAT

MODULE EIGENS
  USE PARS
  REAL(KIND=DP) :: EIGVAL(0:NMAX)
  REAL(KIND=DP) :: EIGVAL_OLD(0:NMAX)
  REAL(KIND=DP) :: EIGVEC(0:NMAX,0:NMAX)
  INTEGER :: EIGORD(0:NMAX)
END MODULE EIGENS

PROGRAM TestModules
  USE PARS
  USE GAUSS
  USE LOGS
  USE LX1D
  USE HRHOMAT
  USE TVUMAT
  USE EIGENS

  IMPLICIT NONE
  
  ! Initialize and test variables here
  BOSC = 1.0
  PRINT *, 'Test BOSC =', BOSC
  
  ! Add further tests as needed for each module

END PROGRAM TestModules