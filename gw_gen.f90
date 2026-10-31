program gw_gen
    implicit none
    integer, parameter :: num_samples = 5000, num_steps = 200
    integer :: i, j
    real :: M_c, t, dt, f, phi, h, t_c
    real :: rand_val
    open(unit=10, file='waveforms.csv', status='replace')
    call random_seed()

    dt = 0.05

    do i = 1, num_samples
        call random_number(rand_val)
        M_c = 10.0 + (rand_val * 40.0) ! Chirp mass from 10 to 50
        write(10, '(F8.2,$)') M_c

        phi = 0.0
        t_c = 15.0 + (50.0 / M_c) ! Time to coalescence depends on mass

        do j = 1, num_steps
            t = j * dt
            if (t < t_c - 0.1) then
                f = (1.0 - t/t_c)**(-0.375) ! Newtonian frequency evolution
                phi = phi + (2.0 * 3.1415926 * f) * dt
                h = (f**0.66) * cos(phi) ! Amplitude grows with frequency
            else
                h = 0.0 ! Cutoff at merger
            end if
            write(10, '(A,F12.4,$)') ',', h
        end do
        write(10, *)
    end do
    
    close(10)
    print *, "Generated 5000 synthetic waveforms in waveforms.csv"
end program gw_gen
