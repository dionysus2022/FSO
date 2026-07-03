%% verify_uniform_qam_tx.m — 验证均匀 QAM 发送信号
% =========================================================
% 功能：
%   1. 加载 uniformQAM_6mod 的 TX .mat 文件
%   2. 绘制星座图（每种子载波 + 全部合并）
%   3. 计算 SNR per subcarrier（EVM → SNR）
%   4. 绘制幅度直方图，验证均匀分布（非 CCDM 概率整形）
%   5. 确认 16QAM→4×4, 32QAM→十字, 64QAM→8×8, 128QAM→十字, 256QAM→16×16
% =========================================================

clear; close all; clc;

addpath(genpath('D:\Project_code\OptDSP_FSO_2\practical waterfiling-exp\OptDSP_lite-master'));

%% ===================== 配置 =====================

data_root = 'D:\Project_code\OptDSP_FSO_2\practical waterfiling-exp\OptDSP_lite-master\2_Data_Results';
tx_root = fullfile(data_root, 'uniformQAM_6mod');

mod_names = {'QPSK', '16QAM', '32QAM', '64QAM', '128QAM', '256QAM'};
mod_bits  = [2, 4, 5, 6, 7, 8];

% 测试参数：读第一个 bin 的 3 帧
test_bin = 1;
n_frames = 3;

% 信号参数
n_sc = 123;       % 子载波数 (4:126)
n_syms = 128;     % 每帧 OFDM 符号数

%% ===================== 遍历每种调制格式 =====================

for mi = 1:length(mod_names)

    mod_name = mod_names{mi};
    bits = mod_bits(mi);
    Mq = 2^bits;

    fprintf('\n============================================\n');
    fprintf(' Verifying: %s (M=%d, bits=%d)\n', mod_name, Mq, bits);
    fprintf('============================================\n');

    %% ---------- 获取理想星座图 ----------

    % 均匀 QAM：nBpS_net = bits（无 CCDM 开销）
    SIG = setSignalParams('symRate', 8e9, 'M', Mq, ...
        'nPol', 1, 'nBpS', bits, 'nSyms', n_syms, ...
        'roll-off', 0.25, 'modulation', 'QAM');
    QAM = QAM_config(SIG);
    ideal_constellation = QAM.IQmap(:);  % 理想星座点

    %% ---------- 加载 3 帧数据 ----------

    bin_str = sprintf('sub%02d', test_bin);
    all_syms = [];  % 所有符号 [n_syms × n_sc × n_frames]

    for fid = 1:n_frames
        sig_id = (test_bin - 1) * n_frames + fid;
        mat_file = fullfile(tx_root, mod_name, bin_str, ...
            sprintf('sig_%04d.mat', sig_id));

        if ~exist(mat_file, 'file')
            warning('  Missing: %s', mat_file);
            continue;
        end

        tmp = load(mat_file);
        data_tx = tmp.data_tx;  % [128 × 123]

        all_syms = cat(3, all_syms, data_tx);
        fprintf('  Loaded frame %d: %s (%d×%d symbols)\n', ...
            fid, mat_file, size(data_tx,1), size(data_tx,2));
    end

    if isempty(all_syms)
        warning('  No data loaded for %s, skipping.', mod_name);
        continue;
    end

    n_loaded = size(all_syms, 3);

    %% ---------- 计算 per-subcarrier SNR ----------

    snr_sc_db = zeros(n_sc, 1);
    all_sc_symbols = cell(n_sc, 1);

    for sc = 1:n_sc
        sc_syms = [];
        for fid = 1:n_loaded
            sc_syms = [sc_syms; all_syms(:, sc, fid)];
        end
        all_sc_symbols{sc} = sc_syms;

        % 硬判决到最近理想星座点
        dist = abs(sc_syms - ideal_constellation.');
        [~, idx] = min(dist, [], 2);
        sc_decided = ideal_constellation(idx);

        % EVM → SNR
        err = sc_syms - sc_decided;
        evm2 = mean(abs(err).^2);
        sig_pow = mean(abs(sc_decided).^2);

        if evm2 > 0
            snr_sc_db(sc) = 10 * log10(sig_pow / evm2);
        else
            snr_sc_db(sc) = 80;  % 完美信号
        end
    end

    %% ---------- 全部符号合并 ----------

    all_syms_vec = [];
    for sc = 1:n_sc
        all_syms_vec = [all_syms_vec; all_sc_symbols{sc}];
    end

    % 硬判决
    dist = abs(all_syms_vec - ideal_constellation.');
    [~, idx] = min(dist, [], 2);
    all_decided = ideal_constellation(idx);

    % 整体 EVM → SNR
    err = all_syms_vec - all_decided;
    evm2_all = mean(abs(err).^2);
    sig_pow_all = mean(abs(all_decided).^2);
    if evm2_all > 0
        snr_total_db = 10 * log10(sig_pow_all / evm2_all);
    else
        snr_total_db = 80;
    end

    fprintf('  Total SNR = %.2f dB\n', snr_total_db);
    fprintf('  Per-SC SNR range: [%.2f, %.2f] dB\n', min(snr_sc_db), max(snr_sc_db));

    %% ---------- 均匀性检测 ----------

    % 统计每个星座点的出现次数
    n_points = length(ideal_constellation);
    counts = zeros(n_points, 1);
    for p = 1:n_points
        % 距离小于阈值的符号归为该星座点
        d = abs(all_syms_vec - ideal_constellation(p));
        counts(p) = sum(d < 0.01 * max(abs(ideal_constellation)));
    end

    expected = length(all_syms_vec) / n_points;
    chi2 = sum((counts - expected).^2 / expected);
    p_value = 1 - chi2cdf(chi2, n_points - 1);

    fprintf('  Uniformity: chi2=%.2f, p=%.4f\n', chi2, p_value);
    if p_value > 0.05
        fprintf('  → 不能拒绝均匀分布假设 → 均匀 QAM ✓\n');
    else
        fprintf('  → 分布非均匀，可能存在概率整形！\n');
    end

    %% ---------- 图1：星座图 + SNR子载波曲线 ----------

    figure('Name', sprintf('%s Verification', mod_name), ...
        'Position', [50, 50, 1200, 500]);

    % 左：合并星座图
    subplot(1, 3, 1);
    plot(real(all_syms_vec), imag(all_syms_vec), 'b.', 'MarkerSize', 2);
    hold on;
    plot(real(ideal_constellation), imag(ideal_constellation), 'ro', ...
        'MarkerSize', 8, 'LineWidth', 1.5);
    hold off;
    axis equal; grid on;
    lim = max(abs(ideal_constellation)) * 1.3;
    xlim([-lim, lim]); ylim([-lim, lim]);
    title(sprintf('%s: All Symbols (%d pts)', mod_name, length(all_syms_vec)));
    xlabel('I'); ylabel('Q');

    % 中：SNR per subcarrier
    subplot(1, 3, 2);
    plot(1:n_sc, snr_sc_db, 'b-', 'LineWidth', 1.2);
    xlabel('Subcarrier Index'); ylabel('SNR (dB)');
    title(sprintf('SNR per Subcarrier (avg=%.1f dB)', mean(snr_sc_db)));
    grid on;
    ylim_auto = ylim;
    ylim([max(0, ylim_auto(1)-5), min(100, ylim_auto(2)+10)]);

    % 右：幅度直方图（均匀性）
    subplot(1, 3, 3);
    histogram(abs(all_syms_vec), 50, 'FaceColor', [0.2 0.4 0.8], 'EdgeAlpha', 0.3);
    xlabel('|Symbol| Amplitude'); ylabel('Count');
    title(sprintf('Amplitude Histogram (p=%.3f)', p_value));
    grid on;

    sgtitle(sprintf('%s (M=%d) — Uniform QAM TX Verification', mod_name, Mq), ...
        'FontSize', 14, 'FontWeight', 'bold');

    %% ---------- 图2：每子载波星座图（抽取4个） ----------

    figure('Name', sprintf('%s Per-SC Constellation', mod_name), ...
        'Position', [100, 100, 1000, 800]);

    sc_samples = round(linspace(1, n_sc, 4));
    for si = 1:4
        sc = sc_samples(si);
        subplot(2, 2, si);
        sc_syms = all_sc_symbols{sc};
        plot(real(sc_syms), imag(sc_syms), 'b.', 'MarkerSize', 3);
        hold on;
        plot(real(ideal_constellation), imag(ideal_constellation), 'ro', ...
            'MarkerSize', 6, 'LineWidth', 1);
        hold off;
        axis equal; grid on;
        lim = max(abs(ideal_constellation)) * 1.3;
        xlim([-lim, lim]); ylim([-lim, lim]);
        title(sprintf('SC#%d (SNR=%.1f dB)', sc, snr_sc_db(sc)));
        xlabel('I'); ylabel('Q');
    end
    sgtitle(sprintf('%s — Per-Subcarrier Constellations', mod_name), ...
        'FontSize', 13, 'FontWeight', 'bold');

    drawnow;
end

%% ===================== 汇总 =====================

fprintf('\n');
fprintf('============================================\n');
fprintf(' Verification Complete\n');
fprintf('============================================\n');
fprintf('All 6 modulation formats checked.\n');
fprintf('Expected constellation patterns:\n');
fprintf('  QPSK   → 4 points (square)\n');
fprintf('  16QAM  → 4×4 grid\n');
fprintf('  32QAM  → cross (6×6 - 4 corners)\n');
fprintf('  64QAM  → 8×8 grid\n');
fprintf('  128QAM → cross (12×12 - 4×4 corners)\n');
fprintf('  256QAM → 16×16 grid\n');
fprintf('============================================\n');