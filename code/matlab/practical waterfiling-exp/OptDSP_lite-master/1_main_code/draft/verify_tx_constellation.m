%% verify_tx_constellation.m - 验证发送信号的星座密度图
clear; close all; clc;

data_root = 'D:\Project_code\OptDSP_FSO_2\practical waterfiling-exp\OptDSP_lite-master\2_Data_Results\tx_1frame_5mod';
mod_names = {'2QAM','4QAM','16QAM','64QAM','256QAM'};
mod_bits  = [1, 2, 4, 6, 8];

samples = {[1,1], [2,2]};

fprintf('正在检查信号文件...\n');
for m = 1:length(mod_names)
    for s = 1:2
        sub_idx = samples{s}(1);
        sig_idx = samples{s}(2);
        mat_path = fullfile(data_root, mod_names{m}, ...
            sprintf('sub%02d', sub_idx), sprintf('sig_%04d.mat', sig_idx));
        
        if ~exist(mat_path, 'file')
            error('文件不存在: %s', mat_path);
        end
        load(mat_path, 'data_tx');
        sym = data_tx(:);
        fprintf('  %s-sub%02d: %d symbols, I=[%.4f,%.4f]\n', ...
            mod_names{m}, sub_idx, length(sym), min(real(sym)), max(real(sym)));
    end
end
fprintf('所有文件加载成功!\n\n');

%% 绘制二维星座密度图 - 每种调制格式单独一个窗口
fprintf('开始绘图...\n');
set(0, 'DefaultFigureVisible', 'on');

for m = 1:length(mod_names)
    figure('Position', [50+(m-1)*80, 50+(m-1)*80, 800, 400], 'Visible', 'on');
    for s = 1:2
        sub_idx = samples{s}(1);
        sig_idx = samples{s}(2);
        mat_path = fullfile(data_root, mod_names{m}, ...
            sprintf('sub%02d', sub_idx), sprintf('sig_%04d.mat', sig_idx));
        load(mat_path, 'data_tx');
        sym = data_tx(:);

        subplot(1, 2, s);
        plot(real(sym), imag(sym), '.', 'MarkerSize', 3);
        axis equal; grid on;
        lim = max(abs([real(sym); imag(sym)])) * 1.1;
        xlim([-lim, lim]); ylim([-lim, lim]);
        title(sprintf('%s - sub%02d/sig_%04d', ...
            mod_names{m}, sub_idx, sig_idx), 'FontSize', 12);
        xlabel('In-Phase'); ylabel('Quadrature');
        set(gca, 'FontSize', 10);
        drawnow;
    end
    sgtitle(sprintf('%s (bits=%d)', mod_names{m}, mod_bits(m)), ...
        'FontSize', 14, 'FontWeight', 'bold');
end
fprintf('绘图完成! 共 %d 个独立窗口。\n', length(mod_names));