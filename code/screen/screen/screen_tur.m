% ==========================================================
% 脚本功能: 生成弱/中/强 三档湍流相位屏 (每档 50 张, 共 150 张)
% 方法: ft_sh_phase_screen (傅里叶 + 次谐波法, 精度高于纯 FFT)
% 保存路径: D:\Projects_python\0508\matlab\screens\
% 文件名: screen_%s_%d_sim.mat  / screen_%s_%d.png
% ==========================================================
clear all;
clc;

% --- 基础参数设置 ---
D = 10e-3;      % 物理边长 (10 mm)
N = 1920;       % 每一边的网格点数 (SLM 宽度分辨率)
L0 = 10;        % 外尺度 (10 m)
l0 = 0.1e-3;    % 内尺度 (0.1 mm)
delta = D/N;    % 网格物理间距

% --- 湍流配置 (三档: 弱/中/强, 每档 20 张) ---
% r0 取值与 one_click_run.m 保持一致
turbulence_configs = struct( ...
    'name',     {'weak',     'moderate', 'strong'}, ...
    'r0',       {3.0e-3,     1.9e-3,     0.8e-3}, ...
    'seed_base',{1000,       2000,       3000});
num_screens_per_group = 50;

% --- 保存路径: 与 one_click_run.m Step2 读取路径一致 ---
save_dir = 'D:\Project_code\FSO_research\code\matlab\screens\';
if ~exist(save_dir, 'dir')
    mkdir(save_dir);
end

fprintf('============================================\n');
fprintf('  生成 150 张相位屏 (弱/中/强 各 50 张)\n');
fprintf('  方法: 傅里叶 + 次谐波法\n');
fprintf('  保存路径: %s\n', save_dir);
fprintf('============================================\n\n');

% --- 核心循环: 生成 3 档 × 20 张 = 60 张 ---
total_count = 0;
for g = 1:3
    group_name = turbulence_configs(g).name;
    r0 = turbulence_configs(g).r0;
    seed_base = turbulence_configs(g).seed_base;
    
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
        filename_png = fullfile(save_dir, sprintf('screen_%s_%d.png', group_name, idx));
        imwrite(bb, filename_png, 'png');
        
        % ==========================================
        % 产出 B: MAT 文件 (高精度浮点相位, 供虚拟信道仿真使用)
        % ==========================================
        filename_mat = fullfile(save_dir, sprintf('screen_%s_%d_sim.mat', group_name, idx));
        save(filename_mat, 'phz_crop');
        
        total_count = total_count + 1;
    end
    
    fprintf('  [%s] 完成 (%d 张)\n', group_name, num_screens_per_group);
end

fprintf('\n============================================\n');
fprintf('  全部生成完毕! 共 %d 张相位屏\n', total_count);
fprintf('  输出目录: %s\n', save_dir);
fprintf('============================================\n');
fprintf('  下一步: 运行 one_click_run.m Step2 (虚拟信道仿真)\n');
fprintf('============================================\n');
