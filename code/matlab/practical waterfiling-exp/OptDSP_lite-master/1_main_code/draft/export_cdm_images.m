% =========================================================================
% export_cdm_images.m - 从实验生成的 .mat 数据集中批量导出纯净 CDM 图像
% =========================================================================
% 功能：遍历 dataset_cdm 下的 weak 和 strong 文件夹
% 将 Distorted_CDM 和 Ideal_CDM 直接映射为 64x64 像素的 PNG 图像
% =========================================================================
clear; close all; clc;

%% 1. 路径配置 (自动对接你的处理代码路径)
data_root    = 'D:\Project_code\OptDSP_FSO_2\practical waterfiling-exp\OptDSP_lite-master\2_Data_Results';
dataset_root = fullfile(data_root, 'dataset_cdm'); 
in_dirs      = {fullfile(dataset_root, 'weak'), fullfile(dataset_root, 'strong')};

% 创建存放最终图片的根目录
out_img_root = fullfile(dataset_root, 'images_png');
if ~exist(out_img_root, 'dir'), mkdir(out_img_root); end

%% 2. 图像保存偏好设置
% 深度学习中，CDM 可以作为单通道灰度图，也可以作为三通道伪彩热力图
% 设定为 true  -> 保存为类似 'jet' 的热力图 (RGB 3通道，适合人眼观察和迁移学习模型)
% 设定为 false -> 保存为纯灰度图 (单通道，严格对应矩阵数值，节约显存)
use_colormap = true; 
cmap = jet(256); % 预生成 256 阶伪彩色表

%% 3. 批量读取与转换
fprintf('\n🚀 开始从 .mat 文件批量生成纯净 CDM 图像...\n');
total_saved = 0;

for d_idx = 1:length(in_dirs)
    current_in_dir = in_dirs{d_idx};
    if ~exist(current_in_dir, 'dir'), continue; end
    
    % 获取当前文件夹名称 (weak 或 strong)
    [~, turb_level, ~] = fileparts(current_in_dir);
    
    % 在图片输出目录建立对应的子文件夹
    out_dir = fullfile(out_img_root, turb_level);
    if ~exist(out_dir, 'dir'), mkdir(out_dir); end
    
    % 获取当前目录下所有 mat 文件
    mat_files = dir(fullfile(current_in_dir, '*.mat'));
    
    for f_idx = 1:length(mat_files)
        mat_name = mat_files(f_idx).name;
        mat_path = fullfile(current_in_dir, mat_name);
        
        % 载入指定的变量
        % 这里包含了你生成的 Distorted_CDM 和 Ideal_CDM
        load(mat_path, 'Distorted_CDM', 'Ideal_CDM', 'Label_Bits');
        
        % 构造基础输出文件名，移除 .mat 后缀
        [~, base_name, ~] = fileparts(mat_name);
        
        % 设置输出路径
        distorted_img_path = fullfile(out_dir, sprintf('%s_Dist_Mod%d.png', base_name, Label_Bits));
        ideal_img_path     = fullfile(out_dir, sprintf('%s_Ideal_Mod%d.png', base_name, Label_Bits));
        
        %% --- 核心图像生成机制：像素级精准保存 ---
        % 注意：因为你的 generate_CDM_Smooth 函数已经做了 cdm / max(cdm(:))，
        % 所以这里的矩阵值已经严格在 [0, 1] 之间，非常适合直接转图像。
        
        if use_colormap
            % 转换为伪彩色 RGB 图像 (尺寸变为 64 x 64 x 3)
            img_distorted = ind2rgb(round(Distorted_CDM * 255), cmap);
            img_ideal     = ind2rgb(round(Ideal_CDM * 255), cmap);
            
            imwrite(img_distorted, distorted_img_path);
            imwrite(img_ideal, ideal_img_path);
        else
            % 直接保存为灰度图 (尺寸为 64 x 64 单通道)
            imwrite(Distorted_CDM, distorted_img_path);
            imwrite(Ideal_CDM, ideal_img_path);
        end
        
        total_saved = total_saved + 1;
        
        % 每 50 个文件打印一次进度
        if mod(total_saved, 50) == 0
            fprintf('  已成功生成并保存 %d 组 CDM 图像...\n', total_saved);
        end
    end
end

fprintf('\n==================================================================\n');
fprintf('  ✅ CDM 图像批量转换圆满成功！\n');
fprintf('  累计处理并导出：%d 组文件（共 %d 张图片，包含畸变与理想图）\n', total_saved, total_saved * 2);
fprintf('  图片存储路径：%s\n', out_img_root);
fprintf('==================================================================\n');
