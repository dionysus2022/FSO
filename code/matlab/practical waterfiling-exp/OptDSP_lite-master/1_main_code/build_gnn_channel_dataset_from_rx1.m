%% 01_build_gnn_channel_dataset_from_rx1.m
% -------------------------------------------------------------------------
% 作用：
%   从 batch_rx1_snr_450frames.m 生成的 rx1_batch_450frames_results 中，
%   提取每一帧的“子载波级残余信道响应 H_res(k)”和“子载波SNR曲线”，
%   生成后续 Python GNN 信道建模所需的数据集。
%
% 重要说明：
%   1) 你当前批处理脚本保存的 rx_sc 已经经过 LTS 信道均衡。
%      因此这里估计的是 residual / effective channel：
%          rx_sc(k,n) ≈ H_res(k) * tx_symbol(k,n) + noise(k,n)
%      它不是完全未经均衡的原始光湍流物理信道 H_raw。
%   2) 对于你的“实验驱动GNN信道增强”路线，这个 residual channel 仍然有价值，
%      因为它包含真实实验系统中的残余幅度畸变、相位扰动、子载波SNR起伏。
%   3) 如果后续要学习更纯粹的原始湍流信道，需要回到批处理脚本中额外保存
%      demod_one_frame_from_lts_start_local 里面的 LTS 估计 H。
%
% 输入：
%   batch_summary.csv
%   每个 .bin 对应的 *_rx1_batch.mat
%
% 输出：
%   gnn_channel_dataset.mat    可被 scipy.io.loadmat 读取的 MATLAB v7 文件
%   gnn_channel_dataset_meta.csv
%   build_log.txt
%
% 使用方式：
%   1. 修改 cfg.batch_root 为你的批处理输出目录。
%   2. 在 MATLAB 中运行本脚本。
%   3. 用 02_train_gnn_channel_vae.py 训练 GNN 信道模型。
% -------------------------------------------------------------------------

clear; clc; close all;

%% ===================== 用户配置区 =====================

cfg = struct();

% 这里改成你 batch_rx1_snr_450frames.m 的输出目录。
cfg.batch_root = 'D:\Project_code\OptDSP_FSO_2\practical waterfiling-exp\OptDSP_lite-master\2_Data_Results\rx1_batch_450frames_results\2026.06.26';

cfg.summary_csv = fullfile(cfg.batch_root, 'batch_summary.csv');

% GNN数据集输出目录
cfg.out_dir = fullfile(cfg.batch_root, 'gnn_channel_dataset');
if ~exist(cfg.out_dir, 'dir')
    mkdir(cfg.out_dir);
end

cfg.out_mat  = fullfile(cfg.out_dir, 'gnn_channel_dataset.mat');
cfg.out_csv  = fullfile(cfg.out_dir, 'gnn_channel_dataset_meta.csv');
cfg.log_file = fullfile(cfg.out_dir, 'build_log.txt');

% SNR质量门限。建议先不要设太高，否则样本会太少。
% 如果你想保留全部成功帧，设为 -Inf。
cfg.min_frame_snr_db = -Inf;

% 是否只保留弱/强湍流 sub01/sub03。
cfg.keep_turbulence = {'weak', 'strong'};

% 如果你现在只想做信道建模，不关心调制是否均衡，可以保留全部调制。
% 如果你只想先用某一种调制估计信道，例如 {'64QAM'}，在这里修改。
cfg.keep_mod_list = {};  % 空 cell 表示不过滤调制格式

% 子载波数量。你的系统 carrier_loc = 4:126，因此默认 123。
cfg.n_sc_expected = 123;

% 每帧最少参与 LS 估计的 OFDM 符号数。
cfg.min_n_sym = 40;

% 是否保存每帧对齐后的 rx/tx 残差统计，不保存完整 rx/tx，避免数据集太大。
cfg.save_extra_quality = true;

% 为了 Python scipy.io.loadmat 兼容，输出使用 -v7，而不是 -v7.3。
cfg.save_version = '-v7';

%% ===================== 日志 =====================

diary(cfg.log_file);
diary on;

fprintf('\n============================================================\n');
fprintf(' Build GNN channel dataset from RX1 batch results\n');
fprintf('============================================================\n');
fprintf('Batch root : %s\n', cfg.batch_root);
fprintf('Summary   : %s\n', cfg.summary_csv);
fprintf('Output MAT: %s\n', cfg.out_mat);
fprintf('Output CSV: %s\n', cfg.out_csv);
fprintf('Min SNR   : %.2f dB\n', cfg.min_frame_snr_db);
fprintf('============================================================\n\n');

if ~exist(cfg.summary_csv, 'file')
    error('Cannot find summary csv: %s', cfg.summary_csv);
end

%% ===================== 读取 summary =====================

T = readtable(cfg.summary_csv, 'TextType', 'string');

required_cols = {'turbulence','sub_name','mod_name','sig_idx','rx_frame_idx', ...
                 'best_tx_id','snr_frame_rx1_db','align_shift','rx_variant','out_file'};

for i = 1:numel(required_cols)
    if ~ismember(required_cols{i}, T.Properties.VariableNames)
        error('Missing column in summary csv: %s', required_cols{i});
    end
end

fprintf('Rows in summary: %d\n', height(T));

% 基础过滤
valid = true(height(T), 1);
valid = valid & isfinite(T.best_tx_id);
valid = valid & isfinite(T.snr_frame_rx1_db);
valid = valid & (T.snr_frame_rx1_db >= cfg.min_frame_snr_db);

if ~isempty(cfg.keep_turbulence)
    valid_turb = false(height(T), 1);
    for i = 1:numel(cfg.keep_turbulence)
        valid_turb = valid_turb | strcmpi(T.turbulence, cfg.keep_turbulence{i});
    end
    valid = valid & valid_turb;
end

if ~isempty(cfg.keep_mod_list)
    valid_mod = false(height(T), 1);
    for i = 1:numel(cfg.keep_mod_list)
        valid_mod = valid_mod | strcmpi(T.mod_name, cfg.keep_mod_list{i});
    end
    valid = valid & valid_mod;
end

T = T(valid, :);
fprintf('Rows after filtering: %d\n', height(T));

if height(T) == 0
    error('No valid rows after filtering. Please check cfg.min_frame_snr_db or summary csv.');
end

%% ===================== 标签映射 =====================

mod_names_all = unique(T.mod_name, 'stable');
turb_names_all = unique(T.turbulence, 'stable');

fprintf('\nTurbulence labels:\n');
for i = 1:numel(turb_names_all)
    fprintf('  %d -> %s\n', i-1, turb_names_all(i));
end

fprintf('\nModulation labels:\n');
for i = 1:numel(mod_names_all)
    fprintf('  %d -> %s\n', i-1, mod_names_all(i));
end

%% ===================== 预分配 =====================

N_max = height(T);
K = cfg.n_sc_expected;

H_re = nan(N_max, K, 'single');
H_im = nan(N_max, K, 'single');
H_abs = nan(N_max, K, 'single');
H_phase = nan(N_max, K, 'single');
snr_sc_db = nan(N_max, K, 'single');
noise_power_sc = nan(N_max, K, 'single');
signal_power_sc = nan(N_max, K, 'single');
evm_sc = nan(N_max, K, 'single');

snr_frame_db = nan(N_max, 1, 'single');
turb_label = nan(N_max, 1, 'single');
mod_label = nan(N_max, 1, 'single');
mod_order = nan(N_max, 1, 'single');
sig_idx_arr = nan(N_max, 1, 'single');
rx_frame_idx_arr = nan(N_max, 1, 'single');
global_frame_idx_arr = nan(N_max, 1, 'single');
best_tx_id_arr = nan(N_max, 1, 'single');
align_shift_arr = nan(N_max, 1, 'single');
n_sym_used_arr = nan(N_max, 1, 'single');

meta_rows = struct([]);

ok_count = 0;
skip_count = 0;
fail_count = 0;

%% ===================== 主循环：逐帧提取 H_res =====================

fprintf('\nStart extracting residual channel per frame...\n');

tic;

for r = 1:height(T)

    if mod(r, 50) == 0 || r == 1 || r == height(T)
        fprintf('[%d/%d] ok=%d skip=%d fail=%d\n', r, height(T), ok_count, skip_count, fail_count);
    end

    try
        out_file = char(T.out_file(r));
        if ~exist(out_file, 'file')
            warning('Missing batch mat: %s', out_file);
            fail_count = fail_count + 1;
            continue;
        end

        S = load(out_file, 'results', 'tx_frame_files', 'mod_name', 'turb', 'sig_idx');

        frame_idx = double(T.rx_frame_idx(r));
        best_tx_id = double(T.best_tx_id(r));

        if frame_idx < 1 || frame_idx > numel(S.results.frame)
            warning('Invalid frame_idx=%g in %s', frame_idx, out_file);
            fail_count = fail_count + 1;
            continue;
        end

        if best_tx_id < 1 || best_tx_id > numel(S.tx_frame_files)
            warning('Invalid best_tx_id=%g in %s', best_tx_id, out_file);
            fail_count = fail_count + 1;
            continue;
        end

        fr = S.results.frame(frame_idx);

        if ~isfield(fr, 'rx_sc') || isempty(fr.rx_sc)
            warning('Empty rx_sc: %s frame=%d', out_file, frame_idx);
            fail_count = fail_count + 1;
            continue;
        end

        rx_sc = double(fr.rx_sc);

        % 使用 summary 中保存的最优 rx_variant，保持和 SNR 匹配一致。
        rx_variant = char(T.rx_variant(r));
        if strcmpi(rx_variant, 'conj')
            rx_sc = conj(rx_sc);
        end

        tx_file = S.tx_frame_files{best_tx_id};
        if ~exist(tx_file, 'file')
            warning('Missing tx frame file: %s', tx_file);
            fail_count = fail_count + 1;
            continue;
        end

        TX = load(tx_file, 'data_tx');
        if ~isfield(TX, 'data_tx')
            warning('No data_tx in %s', tx_file);
            fail_count = fail_count + 1;
            continue;
        end

        tx_ref = double(TX.data_tx.');  % batch脚本中就是这样变成 [123 x 128]

        shift = double(T.align_shift(r));
        if ~isfinite(shift)
            shift = 0;
        end

        [rx_use, tx_use] = align_rx_tx_by_symbol_shift_local(rx_sc, tx_ref, shift);

        if isempty(rx_use) || size(rx_use, 2) < cfg.min_n_sym
            skip_count = skip_count + 1;
            continue;
        end

        [H_sc, snr_sc_this_db, sig_pow, noi_pow, evm_this] = estimate_residual_channel_ls_local(rx_use, tx_use);

        if numel(H_sc) ~= K
            warning('Unexpected n_sc=%d, expected=%d. file=%s', numel(H_sc), K, out_file);
            fail_count = fail_count + 1;
            continue;
        end

        ok_count = ok_count + 1;

        H_re(ok_count, :) = single(real(H_sc(:).'));
        H_im(ok_count, :) = single(imag(H_sc(:).'));
        H_abs(ok_count, :) = single(abs(H_sc(:).'));
        H_phase(ok_count, :) = single(unwrap(angle(H_sc(:))).');
        snr_sc_db(ok_count, :) = single(snr_sc_this_db(:).');
        noise_power_sc(ok_count, :) = single(noi_pow(:).');
        signal_power_sc(ok_count, :) = single(sig_pow(:).');
        evm_sc(ok_count, :) = single(evm_this(:).');

        snr_frame_db(ok_count) = single(10 * log10(mean(10.^(snr_sc_this_db(:)/10)) + eps));
        turb_label(ok_count) = single(find(strcmp(turb_names_all, T.turbulence(r))) - 1);
        mod_label(ok_count) = single(find(strcmp(mod_names_all, T.mod_name(r))) - 1);
        mod_order(ok_count) = single(mod_name_to_order_local(char(T.mod_name(r))));
        sig_idx_arr(ok_count) = single(T.sig_idx(r));
        rx_frame_idx_arr(ok_count) = single(T.rx_frame_idx(r));

        if ismember('global_frame_idx', T.Properties.VariableNames)
            global_frame_idx_arr(ok_count) = single(T.global_frame_idx(r));
        else
            global_frame_idx_arr(ok_count) = single(NaN);
        end

        best_tx_id_arr(ok_count) = single(best_tx_id);
        align_shift_arr(ok_count) = single(shift);
        n_sym_used_arr(ok_count) = single(size(rx_use, 2));

        mr = struct();
        mr.sample_id = ok_count;
        mr.turbulence = string(T.turbulence(r));
        mr.sub_name = string(T.sub_name(r));
        mr.mod_name = string(T.mod_name(r));
        mr.sig_idx = double(T.sig_idx(r));
        mr.rx_frame_idx = double(T.rx_frame_idx(r));
        mr.best_tx_id = best_tx_id;
        mr.align_shift = shift;
        mr.rx_variant = string(rx_variant);
        mr.n_sym_used = size(rx_use, 2);
        mr.snr_frame_db = double(snr_frame_db(ok_count));
        mr.batch_out_file = string(out_file);
        mr.tx_file = string(tx_file);

        if isempty(meta_rows)
            meta_rows = mr;
        else
            meta_rows(end+1, 1) = mr; %#ok<AGROW>
        end

    catch ME
        warning('Failed row %d: %s', r, ME.message);
        fail_count = fail_count + 1;
        continue;
    end
end

elapsed = toc;

fprintf('\nExtraction finished.\n');
fprintf('OK   : %d\n', ok_count);
fprintf('Skip : %d\n', skip_count);
fprintf('Fail : %d\n', fail_count);
fprintf('Time : %.2f sec\n', elapsed);

if ok_count == 0
    error('No valid samples were extracted.');
end

%% ===================== 截断预分配数组 =====================

H_re = H_re(1:ok_count, :);
H_im = H_im(1:ok_count, :);
H_abs = H_abs(1:ok_count, :);
H_phase = H_phase(1:ok_count, :);
snr_sc_db = snr_sc_db(1:ok_count, :);
noise_power_sc = noise_power_sc(1:ok_count, :);
signal_power_sc = signal_power_sc(1:ok_count, :);
evm_sc = evm_sc(1:ok_count, :);

snr_frame_db = snr_frame_db(1:ok_count);
turb_label = turb_label(1:ok_count);
mod_label = mod_label(1:ok_count);
mod_order = mod_order(1:ok_count);
sig_idx_arr = sig_idx_arr(1:ok_count);
rx_frame_idx_arr = rx_frame_idx_arr(1:ok_count);
global_frame_idx_arr = global_frame_idx_arr(1:ok_count);
best_tx_id_arr = best_tx_id_arr(1:ok_count);
align_shift_arr = align_shift_arr(1:ok_count);
n_sym_used_arr = n_sym_used_arr(1:ok_count);

subcarrier_index = single(1:K);
subcarrier_index_norm = single(linspace(-1, 1, K));

% 标签名保存为 cell，Python 读取时不一定方便，所以也同步写 CSV。
turb_names = cellstr(turb_names_all);
mod_names = cellstr(mod_names_all);

meta_table = struct2table(meta_rows);
writetable(meta_table, cfg.out_csv);

%% ===================== 保存 MAT =====================

fprintf('\nSaving dataset...\n');

save(cfg.out_mat, ...
    'H_re', 'H_im', 'H_abs', 'H_phase', ...
    'snr_sc_db', 'noise_power_sc', 'signal_power_sc', 'evm_sc', ...
    'snr_frame_db', 'turb_label', 'mod_label', 'mod_order', ...
    'sig_idx_arr', 'rx_frame_idx_arr', 'global_frame_idx_arr', ...
    'best_tx_id_arr', 'align_shift_arr', 'n_sym_used_arr', ...
    'subcarrier_index', 'subcarrier_index_norm', ...
    'turb_names', 'mod_names', 'cfg', cfg.save_version);

fprintf('Saved MAT: %s\n', cfg.out_mat);
fprintf('Saved CSV: %s\n', cfg.out_csv);

%% ===================== 简单质量统计 =====================

fprintf('\n============================================================\n');
fprintf(' Dataset statistics\n');
fprintf('============================================================\n');
fprintf('Samples       : %d\n', ok_count);
fprintf('Subcarriers   : %d\n', K);
fprintf('Frame SNR mean: %.2f dB\n', mean(double(snr_frame_db), 'omitnan'));
fprintf('Frame SNR std : %.2f dB\n', std(double(snr_frame_db), 'omitnan'));

for i = 1:numel(turb_names_all)
    idx = turb_label == (i-1);
    fprintf('Turb %-8s: %d samples, SNR mean %.2f dB\n', ...
        turb_names_all(i), sum(idx), mean(double(snr_frame_db(idx)), 'omitnan'));
end

for i = 1:numel(mod_names_all)
    idx = mod_label == (i-1);
    fprintf('Mod  %-8s: %d samples, SNR mean %.2f dB\n', ...
        mod_names_all(i), sum(idx), mean(double(snr_frame_db(idx)), 'omitnan'));
end

fprintf('============================================================\n');

diary off;

%% =====================================================================
%% Local helper functions
%% =====================================================================

function [rx_use, tx_use] = align_rx_tx_by_symbol_shift_local(rx_sc, tx_ref, shift)
    rx = rx_sc;
    tx = tx_ref;

    n_sc = min(size(rx,1), size(tx,1));
    rx = rx(1:n_sc, :);
    tx = tx(1:n_sc, :);

    nr = size(rx,2);
    nt = size(tx,2);

    if shift >= 0
        r_start = 1 + shift;
        t_start = 1;
    else
        r_start = 1;
        t_start = 1 - shift;
    end

    n_sym = min(nr - r_start + 1, nt - t_start + 1);

    if n_sym < 20
        rx_use = [];
        tx_use = [];
        return;
    end

    rx_use = rx(:, r_start:r_start+n_sym-1);
    tx_use = tx(:, t_start:t_start+n_sym-1);
end

function [H_sc, snr_sc_db, sig_pow, noi_pow, evm_sc] = estimate_residual_channel_ls_local(rx_use, tx_use)
    % LS估计每个子载波的复数残余信道：
    %   H(k) = argmin_H ||rx(k,:) - H * tx(k,:)||^2
    H_sc = sum(rx_use .* conj(tx_use), 2) ./ (sum(abs(tx_use).^2, 2) + eps);

    rx_hat = H_sc .* tx_use;
    err = rx_use - rx_hat;

    sig_pow = mean(abs(rx_hat).^2, 2);
    noi_pow = mean(abs(err).^2, 2) + eps;

    snr_sc_lin = sig_pow ./ noi_pow;
    snr_sc_db = 10 * log10(snr_sc_lin + eps);

    evm_sc = sqrt(noi_pow ./ (sig_pow + eps));
end

function M = mod_name_to_order_local(mod_name)
    switch upper(strtrim(mod_name))
        case 'QPSK'
            M = 4;
        case '16QAM'
            M = 16;
        case '32QAM'
            M = 32;
        case '64QAM'
            M = 64;
        case '128QAM'
            M = 128;
        case '256QAM'
            M = 256;
        otherwise
            M = NaN;
    end
end
