% example_ft_sh_phase_screen.m
clear all
for t = 20:100
    D = 0.7680; % length of one side of square phase screen [m]
    r0 = 0.064 ; % coherence diameter [m]
    N = 1980; % number of grid points per side
    L0 = 1000; % outer scale [m]
    l0 = 0.0001;% inner scale [m]

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
    filename=['C:\程序与一些功能\大气湍流屏仿真\画大气湍流图\',num2str(t),'.png'];
%     imwrite(aa(1:1080,:),filename,'png')
    imwrite(bb(1:1080,:),filename,'png')
%     imwrite(bb(1:1080,:),num2str(t),'.jpg')
end