% ==========================================================
% 脚本功能：生成 1张 适用于物理光路实验的 0湍流 基线图 (含闪耀光栅)
% ==========================================================
clear all;
clc;

N = 1920;       % 每一边的网格点数 (SLM 宽度分辨率)
save_dir = 'D:\screen\';
if ~exist(save_dir, 'dir')
    mkdir(save_dir);
end

% --- 生成闪耀光栅 (Blazed Grating) 相位 ---
% grating_period 控制光束偏折的角度。
% 如果光束没有打进光阑，可以尝试修改这个值 (通常在 4~20 之间，取决于你的空间布局)
grating_period = 8; 
[X, ~] = meshgrid(1:N, 1:N);

% 生成 0~2pi 的锯齿波相位，这就是光栅
phz_grating = 2 * pi * mod(X, grating_period) / grating_period; 

% 此时你的“0湍流”总相位就仅仅是光栅的相位
phz_total = phz_grating; 
% (注: 如果你的系统还需要加载菲涅尔透镜聚焦，也应该在这里加起来)

% SLM 硬件适配处理：线性映射到 0~1 的灰度区间
bb = phz_total / (2*pi); 

% 裁剪至 1080p 分辨率并保存
filename = [save_dir, 'Zero_Turbulence_Physical.png'];
imwrite(bb(1:1080, :), filename, 'png');

disp('✅ 物理实验版 0湍流基线图 (含闪耀光栅) 生成完毕！');