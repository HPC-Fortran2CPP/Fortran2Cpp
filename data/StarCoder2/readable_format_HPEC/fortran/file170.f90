subroutine  adi


call compute_rhs

call txinvr

call x_solve

call y_solve

call z_solve

call add

return
end
