% ==========================================================
% 脚本功能: 生成强湍流相位屏 200 张 (仅强湍流)
% 方法: ft_sh_phase_screen (傅里叶 + 次谐波法)
% 保存路径: D:\Project_code\FSO_research\code\matlab\screens\
% 子文件夹: screens_强/ (PNG) 和 screens_强_mat/ (MAT)
% 文件名: screen_strong_%d.png  / screen_strong_%d_sim.mat
% ==========================================================
clear all;
clc;

% --- 基础参数设置 (保持不变) ---
D = 10e-3;      % 物理边长 (10 mm)
N = 1920;       % 每一边的网格点数 (SLM 宽度分辨率)
L0 = 10;        % 外尺度 (10 m)
l0 = 0.1e-3;    % 内尺度 (0.1 mm)
delta = D/N;    % 网格物理间距

% --- 强湍流配置 (仅此一档) ---
r0 = 0.8e-3;                % 大气相干直径 (强湍流)
seed_base = 9000;           % 新种子基值，保证与之前不同
num_screens = 200;          % 生成 200 张

% --- 保存路径 ---
root_dir = 'D:\Project_code\FSO_research\code\matlab\screens\';
if ~exist(root_dir, 'dir')
    mkdir(root_dir);
end

% 强湍流专用子文件夹
save_dir_img = fullfile(root_dir, 'screens_强');
save_dir_mat = fullfile(root_dir, 'screens_强_mat');
if ~exist(save_dir_img, 'dir')
    mkdir(save_dir_img);
end
if ~exist(save_dir_mat, 'dir')
    mkdir(save_dir_mat);
end

fprintf('============================================\n');
fprintf('  生成强湍流相位屏 200 张\n');
fprintf('  方法: 傅里叶 + 次谐波法\n');
fprintf('  保存路径: %s\n', root_dir);
fprintf('  r0 = %.1f mm\n', r0*1000);
fprintf('============================================\n\n');

% --- 核心循环: 生成 200 张 ---
total_count = 0;
for idx = 1:num_screens
    seed = seed_base + idx;
    rng(seed);  % 固定随机种子，保证可复现
    
    % 生成随机大气相位屏 (傅里叶 + 次谐波法)
    [phz_lo, phz_hi] = ft_sh_phase_screen(r0, N, delta, L0, l0);
    phz = phz_lo + phz_hi;
    
    % 提取适配 1080p 分辨率的有效区域
    phz_crop = phz(1:1080, :);
    
    % ==========================================
    % 产出 A: PNG 图片 (用于 SLM 硬件加载)
    % ==========================================
    dis = mod(phz_crop, 2*pi);    % 限制在 0~2pi
    bb = mat2gray(dis);           % 映射到 0~1 灰度
    filename_png = fullfile(save_dir_img, sprintf('screen_strong_%d.png', idx));
    imwrite(bb, filename_png, 'png');
    
    % ==========================================
    % 产出 B: MAT 文件 (高精度浮点相位, 供仿真使用)
    % ==========================================
    filename_mat = fullfile(save_dir_mat, sprintf('screen_strong_%d_sim.mat', idx));
    save(filename_mat, 'phz_crop');
    
    total_count = total_count + 1;
    
    if mod(idx, 10) == 0
        fprintf('  已生成 %d/%d 张\n', idx, num_screens);
    end
end

fprintf('\n============================================\n');
fprintf('  生成完毕! 共 %d 张强湍流相位屏\n', total_count);
fprintf('  PNG: %s\n', save_dir_img);
fprintf('  MAT: %s\n', save_dir_mat);
fprintf('============================================\n');