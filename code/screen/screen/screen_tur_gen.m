% ==========================================================
% 脚本功能: 生成极弱/中/强 三档湍流相位屏 (每档 100 张, 共 300 张)最新
% 方法: ft_sh_phase_screen (傅里叶 + 次谐波法, 精度高于纯 FFT)
% 保存路径: D:\Project_code\FSO_research\code\matlab\screens\
% 子文件夹: screens_极弱/ screens_中等/ screens_强/
% 文件名: screen_%s_%d.png  / screen_%s_%d_sim.mat
% ==========================================================
clear all;
clc;

% --- 基础参数设置 ---
D = 10e-3;      % 物理边长 (10 mm)
N = 1920;       % 每一边的网格点数 (SLM 宽度分辨率)
L0 = 10;        % 外尺度 (10 m)
l0 = 0.1e-3;    % 内尺度 (0.1 mm)
delta = D/N;    % 网格物理间距

% --- 湍流配置 (三档: 极弱/中/强, 每档 100 张) ---
% r0 越大湍流越弱 (大气相干直径越大)
turbulence_configs = struct( ...
    'name',     {'极弱',     '中等',     '强'}, ...
    'name_en',  {'very_weak','moderate', 'strong'}, ...
    'r0',       {5.0e-3,     1.9e-3,     0.8e-3}, ...
    'seed_base',{1000,       2000,       3000});
num_screens_per_group = 100;

% --- 保存路径: 根路径 + 子文件夹 ---
root_dir = 'D:\Project_code\FSO_research\code\matlab\screens\';
if ~exist(root_dir, 'dir')
    mkdir(root_dir);
end

fprintf('============================================\n');
fprintf('  生成 300 张相位屏 (极弱/中/强 各 100 张)\n');
fprintf('  方法: 傅里叶 + 次谐波法\n');
fprintf('  保存路径: %s\n', root_dir);
fprintf('============================================\n\n');

% --- 核心循环: 生成 3 档 × 100 张 = 300 张 ---
total_count = 0;
for g = 1:length(turbulence_configs)
    group_name = turbulence_configs(g).name;       % '极弱','中等','强'
    group_name_en = turbulence_configs(g).name_en; % 'very_weak','moderate','strong'
    r0 = turbulence_configs(g).r0;
    seed_base = turbulence_configs(g).seed_base;
    
    % 子文件夹: screens_极弱/ screens_中等/ screens_强/ (PNG)
    %          screens_极弱_mat/ screens_中等_mat/ screens_强_mat/ (MAT)
    save_dir_img = fullfile(root_dir, sprintf('screens_%s', group_name));
    save_dir_mat = fullfile(root_dir, sprintf('screens_%s_mat', group_name));
    if ~exist(save_dir_img, 'dir')
        mkdir(save_dir_img);
    end
    if ~exist(save_dir_mat, 'dir')
        mkdir(save_dir_mat);
    end
    
    fprintf('=== [%s] 组 (r0 = %.1f mm) ===\n', group_name, r0*1000);
    
    for idx = 1:num_screens_per_group
        seed = seed_base + idx;
        rng(seed);  % 固定随机种子, 保证可复现
        
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
        filename_png = fullfile(save_dir_img, sprintf('screen_%s_%d.png', group_name_en, idx));
        imwrite(bb, filename_png, 'png');
        
        % ==========================================
        % 产出 B: MAT 文件 (高精度浮点相位, 供虚拟信道仿真使用)
        % ==========================================
        filename_mat = fullfile(save_dir_mat, sprintf('screen_%s_%d_sim.mat', group_name_en, idx));
        save(filename_mat, 'phz_crop');
        
        total_count = total_count + 1;
        
        if mod(idx, 10) == 0
            fprintf('  [%s] %d/%d\n', group_name, idx, num_screens_per_group);
        end
    end
    
    fprintf('  [%s] 完成 (%d 张)\n', group_name, num_screens_per_group);
end

fprintf('\n============================================\n');
fprintf('  全部生成完毕! 共 %d 张相位屏\n', total_count);
fprintf('  输出目录结构:\n');
for g = 1:length(turbulence_configs)
    group_name = turbulence_configs(g).name;
    fprintf('    PNG: %s\\screens_%s\\  (%d 张)\n', root_dir, group_name, num_screens_per_group);
    fprintf('    MAT: %s\\screens_%s_mat\\  (%d 个)\n', root_dir, group_name, num_screens_per_group);
end
fprintf('============================================\n');
