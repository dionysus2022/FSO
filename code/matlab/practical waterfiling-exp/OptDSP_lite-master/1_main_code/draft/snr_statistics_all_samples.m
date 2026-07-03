%% snr_statistics_all_samples.m
% 全样本 SNR 统计 pipeline
% 依据 snr分析流程.txt 的三层 SNR 结构：
%   Subcarrier-level SNR → Frame-level SNR → File-level SNR
%
% 输入：dataset_paper_pipeline_v2（或 v1）的输出目录
% 输出：SNR 统计 + 诊断图

clear; clear global; close all; clc;

%% ===================== 配置 =====================

% 数据集路径（根据实际输出目录修改）
data_root = 'D:\Project_code\OptDSP_FSO_2\practical waterfiling-exp\OptDSP_lite-master\2_Data_Results';
dataset_name = 'dataset_paper_pipeline_v2';  % 或 'dataset_paper_pipeline_v1'

freq_dir = fullfile(data_root, dataset_name, 'freq_sc');
index_file = fullfile(data_root, dataset_name, 'index.csv');

out_dir = fullfile(data_root, dataset_name, 'snr_analysis');
if ~exist(out_dir, 'dir'), mkdir(out_dir); end

%% ===================== 读取 index.csv =====================

fprintf('Reading index: %s\n', index_file);

if ~exist(index_file, 'file')
    error('index.csv not found: %s', index_file);
end

idx = readtable(index_file, 'Delimiter', ',');

fprintf('Total entries: %d\n', height(idx));

% 筛选有效样本
valid_mask = idx.valid_flag == 1;
fprintf('Valid entries: %d / %d\n', sum(valid_mask), height(idx));

idx_valid = idx(valid_mask, :);

%% ===================== 三层 SNR 统计 =====================

% === 第1层：Subcarrier-level SNR ===
% 每帧每个子载波的 SNR，用于后续诊断图

% === 第2层：Frame-level SNR ===
% 从 index.csv 已有 snr_mean_db 列

frame_snr_db = idx_valid.snr_mean_db;
frame_snr_median = idx_valid.snr_median_db;
frame_valid_ratio = idx_valid.snr_valid_ratio;

fprintf('\n=== Frame-level SNR ===\n');
fprintf('  Mean: %.2f dB\n', nanmean(frame_snr_db));
fprintf('  Median: %.2f dB\n', nanmedian(frame_snr_db));
fprintf('  Std: %.2f dB\n', nanstd(frame_snr_db));
fprintf('  Min: %.2f dB\n', min(frame_snr_db));
fprintf('  Max: %.2f dB\n', max(frame_snr_db));

% === 第3层：File-level SNR（线性平均再转dB）===
% 按 (mod, sub, sig_idx) 分组

[G, mod_name, sub_name, sig_idx_g] = findgroups( ...
    idx_valid.label_name, idx_valid.sub_name, idx_valid.sig_idx);

file_snr_db = splitapply(@(x) 10*log10(mean(10.^(x/10))), frame_snr_db, G);

fprintf('\n=== File-level SNR ===\n');
fprintf('  Total files: %d\n', length(file_snr_db));
fprintf('  Mean: %.2f dB\n', nanmean(file_snr_db));
fprintf('  Median: %.2f dB\n', nanmedian(file_snr_db));
fprintf('  Std: %.2f dB\n', nanstd(file_snr_db));
fprintf('  Min: %.2f dB\n', min(file_snr_db));
fprintf('  Max: %.2f dB\n', max(file_snr_db));

% 按调制格式分组统计
[G2, mod_name2] = findgroups(idx_valid.label_name);
mod_frame_snr = splitapply(@(x) {x}, frame_snr_db, G2);

fprintf('\n=== 按调制格式的 Frame SNR ===\n');
for mi = 1:length(mod_name2)
    snr_m = mod_frame_snr{mi};
    fprintf('  %s: mean=%.2f median=%.2f std=%.2f n=%d\n', ...
        mod_name2{mi}, nanmean(snr_m), nanmedian(snr_m), nanstd(snr_m), length(snr_m));
end

%% ===================== 保存统计结果 =====================

stats = struct();
stats.frame_snr_db = frame_snr_db;
stats.frame_snr_median = frame_snr_median;
stats.frame_valid_ratio = frame_valid_ratio;
stats.file_snr_db = file_snr_db;
stats.n_frames_total = height(idx);
stats.n_frames_valid = sum(valid_mask);
stats.n_files = length(file_snr_db);
stats.frame_mean_db = nanmean(frame_snr_db);
stats.frame_median_db = nanmedian(frame_snr_db);
stats.frame_std_db = nanstd(frame_snr_db);
stats.frame_min_db = min(frame_snr_db);
stats.frame_max_db = max(frame_snr_db);
stats.file_mean_db = nanmean(file_snr_db);
stats.file_median_db = nanmedian(file_snr_db);
stats.file_std_db = nanstd(file_snr_db);
stats.file_min_db = min(file_snr_db);
stats.file_max_db = max(file_snr_db);

save(fullfile(out_dir, 'snr_statistics.mat'), 'stats');
fprintf('\nStatistics saved to: %s\n', fullfile(out_dir, 'snr_statistics.mat'));

%% ===================== 图1：Frame-level SNR 直方图 =====================

figure('Position', [100 100 1200 800]);

subplot(2, 3, 1);
histogram(frame_snr_db, 40, 'FaceColor', '#0072BD', 'EdgeColor', 'none');
xlabel('Frame SNR (dB)');
ylabel('Count');
title(sprintf('Frame SNR Distribution\nmean=%.2f dB, median=%.2f dB', ...
    nanmean(frame_snr_db), nanmedian(frame_snr_db)));
grid on;

% 按调制格式分组的 SNR 直方图
subplot(2, 3, 2);
hold on;
colors = lines(length(mod_name2));
legend_str = {};
for mi = 1:length(mod_name2)
    snr_m = mod_frame_snr{mi};
    histogram(snr_m, 30, 'FaceColor', colors(mi,:), ...
        'EdgeColor', 'none', 'FaceAlpha', 0.6);
    legend_str{end+1} = sprintf('%s (n=%d)', mod_name2{mi}, length(snr_m));
end
xlabel('Frame SNR (dB)');
ylabel('Count');
title('Frame SNR by Modulation');
legend(legend_str, 'Location', 'northwest');
grid on;

%% ===================== 图2：Subcarrier SNR 热图 =====================

subplot(2, 3, 3);
fprintf('\nReading freq_sc files for subcarrier SNR...\n');

% 随机采样部分文件画子载波 SNR
n_sc_samples = min(500, height(idx_valid));
rng(0);
sample_idx = randperm(height(idx_valid), n_sc_samples);

all_sc_snr = [];
for si = 1:length(sample_idx)
    row = idx_valid(sample_idx(si), :);
    freq_file = row.out_freq{1};
    if ~exist(freq_file, 'file'), continue; end
    try
        tmp = load(freq_file);
        rx_sc = tmp.sample_freq.rx_sc;  % [n_sc x n_sym]
        sc_snr = 10*log10(mean(abs(rx_sc).^2, 2));  % 每子载波 SNR
        all_sc_snr = [all_sc_snr, sc_snr(:)]; %#ok<AGROW>
    catch
        continue;
    end
end

if ~isempty(all_sc_snr)
    imagesc(all_sc_snr);
    colorbar;
    xlabel('Sample Index');
    ylabel('Subcarrier Index');
    title(sprintf('Per-Subcarrier SNR (n=%d files)', size(all_sc_snr, 2)));
    colormap('jet');
    set(gca, 'YDir', 'normal');
end

%% ===================== 图3：File SNR vs Modulation =====================

subplot(2, 3, 4);
boxplot(frame_snr_db, idx_valid.label_name, 'Colors', lines(length(mod_name2)));
xlabel('Modulation');
ylabel('Frame SNR (dB)');
title('Frame SNR by Modulation (Boxplot)');
grid on;
xtickangle(45);

%% ===================== 图4：SNR 累积分布 =====================

subplot(2, 3, 5);
[f, x] = ecdf(frame_snr_db);
plot(x, f*100, 'LineWidth', 2);
xlabel('Frame SNR (dB)');
ylabel('Cumulative Percentage (%)');
title('SNR Cumulative Distribution');
grid on;
ylim([0 100]);
% 标记关键点
hold on;
for thresh = [0 5 10 15 20]
    pct = sum(frame_snr_db >= thresh) / length(frame_snr_db) * 100;
    plot(thresh, 100-pct, 'ro', 'MarkerSize', 6);
    text(thresh+0.5, 100-pct, sprintf('%.1f%% > %ddB', 100-pct, thresh), ...
        'FontSize', 8);
end
hold off;

%% ===================== 图5：Valid Ratio Distribution =====================

subplot(2, 3, 6);
histogram(frame_valid_ratio, 30, 'FaceColor', '#D95319', 'EdgeColor', 'none');
xlabel('Valid Subcarrier Ratio');
ylabel('Count');
title('Subcarrier Valid Ratio Distribution');
grid on;
xline(0.9, '--r', 'min threshold');

%% ===================== 保存图片 =====================

saveas(gcf, fullfile(out_dir, 'snr_analysis.png'));
fprintf('Figure saved to: %s\n', fullfile(out_dir, 'snr_analysis.png'));

%% ===================== 导出 CSV =====================

% 按调制格式、湍流强度汇总
fprintf('\n=== SNR 汇总表 ===\n');
fprintf('%-10s %-10s %8s %8s %8s %8s\n', ...
    'Mod', 'Turb', 'Mean', 'Median', 'Std', 'N');
fprintf('%s\n', repmat('-', 1, 55));

mods = unique(idx_valid.label_name);
for mi = 1:length(mods)
    m = mods{mi};
    for ti = 1:3
        turb_names = {'weak', 'moderate', 'strong'};
        t = turb_names{ti};
        mask = strcmp(idx_valid.label_name, m) & strcmp(idx_valid.turbulence, t);
        if sum(mask) == 0, continue; end
        snr_mt = idx_valid.snr_mean_db(mask);
        fprintf('%-10s %-10s %8.2f %8.2f %8.2f %8d\n', ...
            m, t, nanmean(snr_mt), nanmedian(snr_mt), nanstd(snr_mt), sum(mask));
    end
end

fprintf('\n===== SNR Analysis Complete =====\n');
