% example_ft_sh_phase_screen.m
clear all
for t = 1:100
    D = 2; % length of one side of square phase screen [m]
    r0 = 0.02; % coherence diameter [m]
    N = 1980; % number of grid points per side
    L0 = 100; % outer scale [m]
    l0 = 0.01;% inner scale [m]

    delta = D/N; % grid spacing [m]
     % spatial grid
    % x = (-N/2 : N/2-1) * delta;
    % y = x;
    % generate a random draw of an atmospheric phase screen
    [phz_lo,phz_hi] = ft_sh_phase_screen(r0, N, delta, L0, l0);
    phz = phz_lo + phz_hi;

    aa = mat2gray(phz);
%     figure
%     imshow(aa)
    dis = mod(phz,2*pi);
    bb = mat2gray(dis);
%     figure
    imshow(bb)
    imwrite(aa(1:1080,:),'','jpg')
    imwrite(bb(1:1080,:),num2str(t),'.jpg')
end