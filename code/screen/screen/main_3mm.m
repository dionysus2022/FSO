% example_ft_sh_phase_screen.m
clear all
for t = 1:50
    D = 10e-3; % length of one side of square phase screen [m]
    r0 = 3e-3 ; % coherence diameter [m]弱湍流
    N = 1920; % number of grid points per side
    L0 = 10; % outer scale [m]
    l0 = 0.1e-3;% inner scale [m]0.1mm

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
%     imshow(bb)
    filename=['D:\matlab\screen\',num2str(t),'.png'];
%     imwrite(aa(1:1080,:),filename,'png')
    imwrite(bb(1:1080,:),filename,'png')
%     imwrite(bb(1:1080,:),num2str(t),'.jpg')
end