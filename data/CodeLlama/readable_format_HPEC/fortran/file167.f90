subroutine exact( i, j, k, u000ijk )



use lu_data
implicit none

integer i, j, k
double precision u000ijk(*)

integer m
double precision xi, eta, zeta

xi  = ( dble ( i - 1 ) ) / ( nx0 - 1 )
eta  = ( dble ( j - 1 ) ) / ( ny0 - 1 )
zeta = ( dble ( k - 1 ) ) / ( nz - 1 )


do m = 1, 5
u000ijk(m) =  ce(m,1)  &
&        + (ce(m,2)  &
&        + (ce(m,5)  &
&        + (ce(m,8)  &
&        +  ce(m,11) * xi) * xi) * xi) * xi  &
&        + (ce(m,3)  &
&        + (ce(m,6)  &
&        + (ce(m,9)  &
&        +  ce(m,12) * eta) * eta) * eta) * eta  &
&        + (ce(m,4)  &
&        + (ce(m,7)  &
&        + (ce(m,10)  &
&        +  ce(m,13) * zeta) * zeta) * zeta) * zeta
end do

return
end
