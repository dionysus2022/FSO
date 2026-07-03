%% verify_uniform_qam_tx_v2.m
% 验证均匀 QAM TX 数据：
%   1) 检查每种调制的星座点是否正确
%   2) 检查 data_tx 是否与缩放后的理想星座匹配
%   3) 检查星座点出现频率是否接近均匀分布
%   4) 输出图和 summary CSV
%
% 注意：
%   tx_uniform_qam_6mod_v2_3frame_bin.m 中 data_tx = carrier_scale * S.tx
%   因此这里必须使用 ideal_constellation * carrier_scale 进行验证。

clear; close all; clc;

addpath(genpath('D:\Project_code\OptDSP_FSO_2\practical waterfiling-exp\OptDSP_lite-master'));

%% ===================== Config =====================

data_root = 'D:\Project_code\OptDSP_FSO_2\practical waterfiling-exp\OptDSP_lite-master\2_Data_Results';
tx_root = fullfile(data_root, 'uniformQAM_6mod_tx_v2');
report_dir = fullfile(tx_root, '_verification_report');

if ~exist(report_dir, 'dir')
    mkdir(report_dir);
end

mod_names = {'QPSK', '16QAM', '32QAM', '64QAM', '128QAM', '256QAM'};
mod_bits  = [2, 4, 5, 6, 7, 8];

test_bin = 1;
n_sc_expected = 123;
n_syms_expected = 128;
carrier_scale = 1 / sqrt(512);

summary = {};

%% ===================== Verify each modulation =====================

for mi = 1:length(mod_names)

    mod_name = mod_names{mi};
    bits = mod_bits(mi);
    Mq = 2^bits;

    fprintf('\n============================================\n');
    fprintf('Verifying %s, M=%d\n', mod_name, Mq);
    fprintf('============================================\n');

    %% ---------- Ideal constellation ----------

    SIG = setSignalParams('symRate', 8e9, 'M', Mq, ...
        'nPol', 1, 'nBpS', bits, 'nSyms', n_syms_expected, ...
        'roll-off', 0.25, 'modulation', 'QAM');

    QAM = QAM_config(SIG);
    ideal_constellation_raw = QAM.IQmap(:);
    ideal_constellation = carrier_scale * ideal_constellation_raw;

    %% ---------- Load combined TX mat ----------

    bin_dir = fullfile(tx_root, mod_name, sprintf('bin_%04d', test_bin));
    mat_file = fullfile(bin_dir, sprintf('sig_%04d.mat', test_bin));

    if ~exist(mat_file, 'file')
        warning('Missing file: %s', mat_file);
        continue;
    end

    tmp = load(mat_file);

    if ~isfield(tmp, 'data_tx_3frame')
        warning('No data_tx_3frame in %s', mat_file);
        continue;
    end

    data_tx_3frame = tmp.data_tx_3frame;
    [n_syms, n_sc, n_frames] = size(data_tx_3frame);

    fprintf('Loaded: %s\n', mat_file);
    fprintf('data_tx_3frame size = [%d, %d, %d]\n', n_syms, n_sc, n_frames);

    if n_sc ~= n_sc_expected || n_syms ~= n_syms_expected
        warning('Unexpected data_tx size. Expected [%d,%d], got [%d,%d].', ...
            n_syms_expected, n_sc_expected, n_syms, n_sc);
    end

    all_syms_vec = data_tx_3frame(:);

    %% ---------- Nearest constellation mapping ----------

    [idx, dmin] = nearest_constellation(all_syms_vec, ideal_constellation);
    decided = ideal_constellation(idx);

    max_err = max(dmin);
    mean_err = mean(dmin);

    err = all_syms_vec - decided;
    evm2 = mean(abs(err).^2);
    sig_pow = mean(abs(decided).^2);

    if evm2 > 0
        evm_snr_db = 10 * log10(sig_pow / evm2);
    else
        evm_snr_db = 300;
    end

    fprintf('Max distance to nearest ideal point = %.3e\n', max_err);
    fprintf('Mean distance to nearest ideal point = %.3e\n', mean_err);
    fprintf('TX self-EVM SNR = %.2f dB\n', evm_snr_db);

    %% ---------- Uniformity test ----------

    counts = accumarray(idx, 1, [length(ideal_constellation), 1]);
    expected = length(all_syms_vec) / length(ideal_constellation);

    chi2_stat = sum((counts - expected).^2 / expected);
    df = length(ideal_constellation) - 1;

    % p = 1 - chi2cdf(chi2, df)，使用 gammainc 避免依赖 Statistics Toolbox
    p_value = gammainc(chi2_stat/2, df/2, 'upper');

    count_min = min(counts);
    count_max = max(counts);
    count_cv = std(counts) / mean(counts);

    fprintf('Uniformity chi2 = %.2f, df=%d, p=%.4g\n', chi2_stat, df, p_value);
    fprintf('Counts: min=%d, max=%d, CV=%.4f, expected=%.1f\n', ...
        count_min, count_max, count_cv, expected);

    if p_value > 0.01
        fprintf('Result: cannot reject uniform constellation sampling. ✓\n');
    else
        fprintf('Result: distribution deviates from ideal uniformity. Check sample size/seed/mapping.\n');
    end

    %% ---------- Per-subcarrier self EVM ----------

    snr_sc_db = zeros(n_sc, 1);
    for sc = 1:n_sc
        sc_syms = squeeze(data_tx_3frame(:, sc, :));
        sc_syms = sc_syms(:);
        [idx_sc, ~] = nearest_constellation(sc_syms, ideal_constellation);
        decided_sc = ideal_constellation(idx_sc);
        e_sc = sc_syms - decided_sc;
        evm2_sc = mean(abs(e_sc).^2);
        sig_pow_sc = mean(abs(decided_sc).^2);
        if evm2_sc > 0
            snr_sc_db(sc) = 10 * log10(sig_pow_sc / evm2_sc);
        else
            snr_sc_db(sc) = 300;
        end
    end

    %% ---------- Save figures ----------

    fig1 = figure('Name', sprintf('%s Uniform TX Verification', mod_name), ...
        'Position', [50, 50, 1300, 420]);

    subplot(1,3,1);
    plot(real(all_syms_vec), imag(all_syms_vec), 'b.', 'MarkerSize', 2);
    hold on;
    plot(real(ideal_constellation), imag(ideal_constellation), 'ro', ...
        'MarkerSize', 7, 'LineWidth', 1.2);
    hold off; axis equal; grid on;
    lim = max(abs(ideal_constellation)) * 1.35;
    xlim([-lim, lim]); ylim([-lim, lim]);
    title(sprintf('%s TX constellation', mod_name));
    xlabel('I'); ylabel('Q');

    subplot(1,3,2);
    bar(counts);
    grid on;
    xlabel('Constellation point index'); ylabel('Count');
    title(sprintf('Uniformity: p=%.3g, CV=%.3f', p_value, count_cv));

    subplot(1,3,3);
    plot(1:n_sc, snr_sc_db, 'b-', 'LineWidth', 1.2);
    grid on;
    xlabel('Subcarrier index'); ylabel('Self-EVM SNR (dB)');
    title(sprintf('Self-EVM SNR, mean=%.1f dB', mean(snr_sc_db)));

    sgtitle(sprintf('%s uniform QAM TX verification', mod_name), ...
        'FontSize', 13, 'FontWeight', 'bold');

    saveas(fig1, fullfile(report_dir, sprintf('%s_verify.png', mod_name)));
    close(fig1);

    %% ---------- Save per-mod counts ----------

    count_table = table((1:length(counts)).', real(ideal_constellation), imag(ideal_constellation), ...
        counts, counts / sum(counts), ...
        'VariableNames', {'point_index', 'I', 'Q', 'count', 'prob'});
    writetable(count_table, fullfile(report_dir, sprintf('%s_constellation_counts.csv', mod_name)));

    %% ---------- Append summary ----------

    summary(end+1, :) = {mod_name, Mq, n_frames, length(all_syms_vec), ...
        max_err, mean_err, evm_snr_db, chi2_stat, df, p_value, ...
        count_min, count_max, count_cv, mean(abs(all_syms_vec).^2)}; %#ok<SAGROW>
end

%% ===================== Summary CSV =====================

summary_table = cell2table(summary, 'VariableNames', { ...
    'mod_name', 'M', 'n_frames', 'n_symbols_total', ...
    'max_err_to_ideal', 'mean_err_to_ideal', 'self_evm_snr_db', ...
    'chi2_stat', 'chi2_df', 'uniformity_p_value', ...
    'count_min', 'count_max', 'count_cv', 'avg_power'});

summary_csv = fullfile(report_dir, 'uniform_qam_tx_verification_summary.csv');
writetable(summary_table, summary_csv);

fprintf('\n============================================\n');
fprintf('Verification complete.\n');
fprintf('Report directory:\n  %s\n', report_dir);
fprintf('Summary CSV:\n  %s\n', summary_csv);
fprintf('============================================\n');

%% ===================== Local function =====================

function [idx, dmin] = nearest_constellation(x, const)
    x = x(:);
    const = const(:).';
    D = abs(bsxfun(@minus, x, const));
    [dmin, idx] = min(D, [], 2);
end
