%% batch_iterative_rx1_dataset_builder_v1.m
% =========================================================================
% 基于 iterative_rx1_style_pipeline.m 的全样本数据集生成脚本
%
% 核心原则：
%   1) 沿用已通过单样本测试的 rx1-style 单帧同步逻辑：
%        packet_edge_power_dect + rx_fine_time_sync_cross_corr
%   2) 不再调用 deOFDM 黑箱，找到 LTS 起点后直接手动解调
%   3) 批量遍历全部 mod/sub/*.bin
%   4) 保存三类特征：
%        time_32768 / freq_sc / cdm_64
%   5) 输出：
%        dataset_index.csv
%        split_index.csv
%        failure_log.csv
%        snr_per_file.csv
%        snr_summary_evm.mat
%        dataset_snr_summary_evm.png
%
% 注意：
%   - EVM-SNR / TX参考匹配仅用于离线数据质量标注与SNR统计；
%   - 后续Python识别模型只允许使用 rx_time / rx_sc / cdm64，不得使用 data_tx、SNR、EVM、best_tx_id 作为输入。
% =========================================================================

clear; clear global; close all; clc;

addpath(genpath('D:\Project_code\OptDSP_FSO_2\practical waterfiling-exp\OptDSP_lite-master'));

global PROG;
PROG.showMessagesLevel = 0;
initProg();

%% ===================== 1. Config =====================

cfg = struct();

cfg.data_root = 'D:\Project_code\OptDSP_FSO_2\practical waterfiling-exp\OptDSP_lite-master\2_Data_Results';
cfg.rx_date   = '2026.06.26';
cfg.tx_root   = fullfile(cfg.data_root, 'tx_3frame_6mod');
cfg.out_root  = fullfile(cfg.data_root, 'dataset_iterative_rx1_final_v1');

cfg.Fs_rx   = 80e9;
cfg.Fs_base = 16e9;

cfg.mod_names = {'QPSK','16QAM','32QAM','64QAM','128QAM','256QAM'};
cfg.sub_list  = {'sub1','sub2','sub3'};

cfg.turb_map = containers.Map('KeyType','char','ValueType','char');
cfg.turb_map('sub1') = 'weak';
cfg.turb_map('sub2') = 'moderate';
cfg.turb_map('sub3') = 'strong';

cfg.n_frames = 3;
cfg.M_time = 32768;

% OFDM structure
cfg.n_syms = 128;
cfg.zeros_head = 80;
cfg.n_fft = 256;
cfg.n_guard = 16;
cfg.sym_len = cfg.n_fft + cfg.n_guard;
cfg.carrier_loc = 4:126;
cfg.n_sc = length(cfg.carrier_loc);
cfg.carrier_loc_demo = [4:126, 132:254];

cfg.header_len_16 = cfg.zeros_head + cfg.n_guard + 2*cfg.n_fft;
cfg.frame_len_16 = cfg.header_len_16 + cfg.sym_len * cfg.n_syms;  % 35424

% CDM
cfg.cdm_bins = 64;
cfg.cdm_clip = 3.0;

% Quality criteria
cfg.min_ofdm_symbols = 100;
cfg.min_snr_evm_db = -5;
cfg.min_snr_sc_valid_ratio = 0.80;
cfg.min_match_margin_db = 0.0;  % 只记录，不强制筛除

% Train/val/test split
cfg.random_seed = 2026;
cfg.train_ratio = 0.70;
cfg.val_ratio = 0.15;
cfg.test_ratio = 0.15;

% Debug/progress
cfg.print_every = 10;

%% ===================== 2. Output dirs =====================

dirs = struct();
dirs.full_frame = fullfile(cfg.out_root, 'full_frame_16G');
dirs.time       = fullfile(cfg.out_root, sprintf('time_%d', cfg.M_time));
dirs.freq       = fullfile(cfg.out_root, 'freq_sc');
dirs.cdm        = fullfile(cfg.out_root, sprintf('cdm_%d', cfg.cdm_bins));
dirs.logs       = fullfile(cfg.out_root, 'logs');
dirs.snr        = fullfile(cfg.out_root, 'snr_results');

make_dir_local(cfg.out_root);
make_dir_local(dirs.full_frame);
make_dir_local(dirs.time);
make_dir_local(dirs.freq);
make_dir_local(dirs.cdm);
make_dir_local(dirs.logs);
make_dir_local(dirs.snr);

%% ===================== 3. Accumulators =====================

index_rows = {};
failure_rows = {};
file_snr_rows = {};

all_frame_snr_evm = [];
all_file_snr_evm = [];
all_sc_snr_evm = [];

all_mod_labels = {};
all_sub_labels = {};
all_turb_labels = {};

stats = struct();
stats.total_files = 0;
stats.ok_read = 0;
stats.fail_read = 0;
stats.ok_tx_ref = 0;
stats.fail_tx_ref = 0;
stats.ok_extract = 0;
stats.fail_extract = 0;
stats.ok_saved = 0;

%% ===================== 4. Main batch loop =====================

for mi = 1:length(cfg.mod_names)

    mod_name = cfg.mod_names{mi};
    label_id = mi - 1;
    [Mq, bits] = mod_to_order_bits_local(mod_name);

    rx_mod_dir = fullfile(cfg.data_root, 'rx_data', cfg.rx_date, mod_name);

    if ~exist(rx_mod_dir, 'dir')
        fprintf('[Skip] Missing modulation folder: %s\n', rx_mod_dir);
        continue;
    end

    for si = 1:length(cfg.sub_list)

        sub_name = cfg.sub_list{si};

        if isKey(cfg.turb_map, sub_name)
            turb_name = cfg.turb_map(sub_name);
        else
            turb_name = sub_name;
        end

        rx_dir = fullfile(rx_mod_dir, sub_name);

        if ~exist(rx_dir, 'dir')
            fprintf('[Skip] Missing sub folder: %s\n', rx_dir);
            continue;
        end

        bin_list = dir(fullfile(rx_dir, '*.bin'));

        if isempty(bin_list)
            fprintf('[Skip] No .bin files: %s\n', rx_dir);
            continue;
        end

        make_dir_local(fullfile(dirs.full_frame, mod_name, sub_name));
        make_dir_local(fullfile(dirs.time, mod_name, sub_name));
        make_dir_local(fullfile(dirs.freq, mod_name, sub_name));
        make_dir_local(fullfile(dirs.cdm, mod_name, sub_name));

        fprintf('\n========== %s / %s : %d files ==========\n', ...
            mod_name, sub_name, length(bin_list));

        for bi = 1:length(bin_list)

            [~, fname] = fileparts(bin_list(bi).name);
            sig_idx = str2double(fname);

            if isnan(sig_idx) || sig_idx < 1
                continue;
            end

            stats.total_files = stats.total_files + 1;

            rx_bin = fullfile(rx_dir, bin_list(bi).name);
            file_key = sprintf('%s_%s_sig%04d', mod_name, sub_name, sig_idx);

            %% ---------- A. Load TX refs ----------
            tx_refs = cell(1, cfg.n_frames);
            ref_ok = true;
            ref_msg = '';

            for tid = 1:cfg.n_frames
                ref_file = fullfile(cfg.tx_root, mod_name, sub_name, ...
                    sprintf('sig_%04d_frame%d.mat', sig_idx, tid));

                if ~exist(ref_file, 'file')
                    ref_ok = false;
                    ref_msg = ['missing TX ref: ' ref_file];
                    break;
                end

                tmp_ref = load(ref_file);

                if ~isfield(tmp_ref, 'data_tx')
                    ref_ok = false;
                    ref_msg = ['TX ref has no data_tx: ' ref_file];
                    break;
                end

                tx_refs{tid} = tmp_ref.data_tx.';  % [123 × 128]
            end

            if ~ref_ok
                stats.fail_tx_ref = stats.fail_tx_ref + 1;
                failure_rows(end+1,:) = {mod_name, sub_name, turb_name, ...
                    sig_idx, 0, 'tx_ref', ref_msg, rx_bin};
                continue;
            else
                stats.ok_tx_ref = stats.ok_tx_ref + 1;
            end

            %% ---------- B. Read RX bin ----------
            try
                [rx80, read_info] = read_keysight_bin_robust_local(rx_bin);

                rx80 = rx80(:).';
                rx80 = rx80 - mean(rx80);
                rx80 = rx80 ./ (rms(rx80) + eps);

                stats.ok_read = stats.ok_read + 1;
            catch ME
                stats.fail_read = stats.fail_read + 1;

                failure_rows(end+1,:) = {mod_name, sub_name, turb_name, ...
                    sig_idx, 0, 'read_bin', ME.message, rx_bin};
                continue;
            end

            %% ---------- C. Resample to 16G ----------
            try
                rx16 = resample(rx80, cfg.Fs_base, cfg.Fs_rx);
                rx16 = rx16(:).';
                rx16 = rx16 - mean(rx16);
                rx16 = rx16 ./ (mean(abs(rx16)) + eps);

                % AWG循环保护：防止第三帧跨采集末尾
                wrap_len = min(length(rx16), 3 * cfg.frame_len_16);
                rx16_ext = [rx16, rx16(1:wrap_len)];
            catch ME
                stats.fail_extract = stats.fail_extract + cfg.n_frames;
                failure_rows(end+1,:) = {mod_name, sub_name, turb_name, ...
                    sig_idx, 0, 'resample', ME.message, rx_bin};
                continue;
            end

            %% ---------- D. Iterative rx1-style extraction ----------
            cursor = 1;
            file_frame_snr = [];
            file_sc_snr_mat = [];

            for rk = 1:cfg.n_frames

                try
                    search_sig = rx16_ext(cursor:end);

                    if length(search_sig) < cfg.frame_len_16
                        error('remaining sequence too short: len=%d', length(search_sig));
                    end

                    [lts_start_rel, frame_start_rel, sync_info] = ...
                        find_one_frame_start_rx1_style_local( ...
                        search_sig, cfg);

                    lts_start_abs   = cursor + lts_start_rel - 1;
                    frame_start_abs = cursor + frame_start_rel - 1;

                    % 从 LTS 起点直接解调，不调用 deOFDM
                    [rx_sc, rx_time_payload, rx_frame16_lts, demod_info] = ...
                        demod_one_frame_from_lts_start_local( ...
                        rx16_ext, lts_start_abs, cfg);

                    % 保存完整时域帧
                    frame_end_abs = frame_start_abs + cfg.frame_len_16 - 1;

                    if frame_start_abs < 1
                        frame_start_abs = 1;
                    end

                    if frame_end_abs > length(rx16_ext)
                        error('frame crosses rx16_ext boundary: start=%d end=%d len=%d', ...
                            frame_start_abs, frame_end_abs, length(rx16_ext));
                    end

                    rx_frame16_full = rx16_ext(frame_start_abs:frame_end_abs);

                    % 三帧TX参考自动匹配 + EVM-SNR
                    [best_tx_id, snr_frame_evm_db, snr_sc_evm_db, ...
                        snr_list_db, snr_match_margin_db] = ...
                        match_rx_to_tx_by_evm_snr_local(rx_sc, tx_refs, cfg);

                    % 辅助功率型SNR/SQI
                    snr_power_frame_db = 10 * log10(mean(abs(rx_sc(:)).^2) + eps);

                    valid_sc = snr_sc_evm_db(isfinite(snr_sc_evm_db));
                    snr_sc_valid_ratio = length(valid_sc) / length(snr_sc_evm_db);

                    if isempty(valid_sc)
                        snr_sc_mean_db = NaN;
                        snr_sc_std_db  = NaN;
                        snr_sc_min_db  = NaN;
                        snr_sc_max_db  = NaN;
                    else
                        snr_sc_mean_db = mean(valid_sc);
                        snr_sc_std_db  = std(valid_sc);
                        snr_sc_min_db  = min(valid_sc);
                        snr_sc_max_db  = max(valid_sc);
                    end

                    valid_clean = ...
                        demod_info.n_use >= cfg.min_ofdm_symbols && ...
                        isfinite(snr_frame_evm_db) && ...
                        snr_frame_evm_db >= cfg.min_snr_evm_db && ...
                        snr_sc_valid_ratio >= cfg.min_snr_sc_valid_ratio;

                    if valid_clean
                        valid_flag = 1;
                        valid_reason = 'valid_clean';
                    else
                        valid_flag = 0;
                        valid_reason = 'valid_but_low_quality';
                    end

                    cdm64 = make_cdm_from_rxsc_local(rx_sc, ...
                        cfg.cdm_bins, cfg.cdm_clip);

                    %% ---------- Save ----------
                    base_name = sprintf('sig_%04d_rxframe%d', sig_idx, rk);

                    full_path = fullfile(dirs.full_frame, mod_name, sub_name, ...
                        [base_name '.mat']);
                    time_path = fullfile(dirs.time, mod_name, sub_name, ...
                        [base_name '.mat']);
                    freq_path = fullfile(dirs.freq, mod_name, sub_name, ...
                        [base_name '.mat']);
                    cdm_path = fullfile(dirs.cdm, mod_name, sub_name, ...
                        [base_name '.mat']);

                    sample = struct();
                    sample.rx_frame_16_full = single(rx_frame16_full);
                    sample.rx_frame_16_lts_aligned = single(rx_frame16_lts);
                    sample.rx_time = single(rx_time_payload);
                    sample.rx_sc = single(rx_sc);
                    sample.cdm64 = single(cdm64);

                    sample.snr_sc_evm_db = single(snr_sc_evm_db);
                    sample.snr_frame_evm_db = snr_frame_evm_db;
                    sample.snr_power_frame_db = snr_power_frame_db;
                    sample.snr_list_db = single(snr_list_db);
                    sample.snr_match_margin_db = snr_match_margin_db;

                    sample.best_tx_frame_id = best_tx_id;
                    sample.mod_name = mod_name;
                    sample.label_id = label_id;
                    sample.mod_order = Mq;
                    sample.bits = bits;
                    sample.sub_name = sub_name;
                    sample.turbulence = turb_name;
                    sample.sig_idx = sig_idx;
                    sample.rx_frame_idx = rk;
                    sample.file_key = file_key;
                    sample.rx_bin_file = rx_bin;
                    sample.frame_start_abs_16 = frame_start_abs;
                    sample.lts_start_abs_16 = lts_start_abs;
                    sample.frame_len_16 = cfg.frame_len_16;
                    sample.sync_info = sync_info;
                    sample.demod_info = demod_info;
                    sample.read_info = read_info;
                    sample.valid_flag = valid_flag;
                    sample.valid_reason = valid_reason;

                    save(full_path, 'sample', '-v7.3');

                    sample_time = struct();
                    sample_time.rx_time = single(rx_time_payload);
                    sample_time.label_id = label_id;
                    sample_time.label_name = mod_name;
                    sample_time.file_key = file_key;
                    sample_time.sig_idx = sig_idx;
                    sample_time.rx_frame_idx = rk;
                    sample_time.snr_frame_evm_db = snr_frame_evm_db;
                    sample_time.valid_flag = valid_flag;
                    save(time_path, 'sample_time', '-v7.3');

                    sample_freq = struct();
                    sample_freq.rx_sc = single(rx_sc);
                    sample_freq.label_id = label_id;
                    sample_freq.label_name = mod_name;
                    sample_freq.file_key = file_key;
                    sample_freq.sig_idx = sig_idx;
                    sample_freq.rx_frame_idx = rk;
                    sample_freq.snr_frame_evm_db = snr_frame_evm_db;
                    sample_freq.valid_flag = valid_flag;
                    save(freq_path, 'sample_freq', '-v7.3');

                    sample_cdm = struct();
                    sample_cdm.cdm64 = single(cdm64);
                    sample_cdm.label_id = label_id;
                    sample_cdm.label_name = mod_name;
                    sample_cdm.file_key = file_key;
                    sample_cdm.sig_idx = sig_idx;
                    sample_cdm.rx_frame_idx = rk;
                    sample_cdm.snr_frame_evm_db = snr_frame_evm_db;
                    sample_cdm.valid_flag = valid_flag;
                    save(cdm_path, 'sample_cdm', '-v7.3');

                    %% ---------- Index ----------
                    index_rows(end+1,:) = { ...
                        full_path, time_path, freq_path, cdm_path, ...
                        file_key, mod_name, label_id, Mq, sub_name, turb_name, ...
                        sig_idx, rk, best_tx_id, ...
                        snr_frame_evm_db, snr_power_frame_db, ...
                        snr_sc_mean_db, snr_sc_std_db, snr_sc_min_db, snr_sc_max_db, ...
                        snr_sc_valid_ratio, snr_match_margin_db, ...
                        demod_info.n_use, demod_info.cfo, ...
                        frame_start_abs, lts_start_abs, ...
                        sync_info.edge_index, sync_info.fine_time_est, ...
                        valid_flag, valid_reason };

                    file_frame_snr(end+1) = snr_frame_evm_db;
                    file_sc_snr_mat = [file_sc_snr_mat, snr_sc_evm_db(:)];

                    all_frame_snr_evm(end+1) = snr_frame_evm_db;
                    all_sc_snr_evm = [all_sc_snr_evm, snr_sc_evm_db(:)];
                    all_mod_labels{end+1} = mod_name;
                    all_sub_labels{end+1} = sub_name;
                    all_turb_labels{end+1} = turb_name;

                    stats.ok_extract = stats.ok_extract + 1;
                    stats.ok_saved = stats.ok_saved + 1;

                    % cursor推进到当前完整帧之后，继续寻找下一帧
                    cursor = frame_start_abs + cfg.frame_len_16;

                catch ME
                    stats.fail_extract = stats.fail_extract + 1;

                    failure_rows(end+1,:) = {mod_name, sub_name, turb_name, ...
                        sig_idx, rk, 'extract_demod_save', ME.message, rx_bin};

                    % 防止卡死：向前推进半帧再继续
                    cursor = cursor + round(0.5 * cfg.frame_len_16);
                end
            end

            %% ---------- File-level SNR ----------
            valid_file_snr = file_frame_snr(isfinite(file_frame_snr));

            if ~isempty(valid_file_snr)
                file_snr_evm_db = 10 * log10(mean(10.^(valid_file_snr / 10)));
                all_file_snr_evm(end+1) = file_snr_evm_db;

                valid_file_sc = file_sc_snr_mat(isfinite(file_sc_snr_mat));

                if isempty(valid_file_sc)
                    file_sc_mean_db = NaN;
                    file_sc_std_db = NaN;
                else
                    file_sc_mean_db = mean(valid_file_sc);
                    file_sc_std_db = std(valid_file_sc);
                end

                file_snr_rows(end+1,:) = {mod_name, sub_name, turb_name, ...
                    sig_idx, file_key, file_snr_evm_db, ...
                    mean(valid_file_snr), median(valid_file_snr), ...
                    std(valid_file_snr), length(valid_file_snr), ...
                    file_sc_mean_db, file_sc_std_db};
            end

            if mod(bi, cfg.print_every) == 0 || bi == length(bin_list)
                fprintf('%s/%s: %d/%d files | readOK=%d | savedFrames=%d | failExtract=%d\n', ...
                    mod_name, sub_name, bi, length(bin_list), ...
                    stats.ok_read, stats.ok_saved, stats.fail_extract);
            end
        end
    end
end

%% ===================== 5. Build output tables =====================

index_varnames = {'FullFramePath','TimePath','FreqPath','CDMPath', ...
    'FileKey','Mod','LabelID','ModOrder','Sub','Turbulence', ...
    'SigIdx','RxFrameIdx','BestTxFrameID', ...
    'SNRFrameEVM_dB','SNRFramePower_dB', ...
    'SNRScMean_dB','SNRScStd_dB','SNRScMin_dB','SNRScMax_dB', ...
    'SNRScValidRatio','SNRMatchMargin_dB', ...
    'NOFDMSymbols','CFO', ...
    'FrameStartAbs16','LTSStartAbs16', ...
    'EdgeIndex','FineTimeEst', ...
    'ValidFlag','ValidReason'};

if ~isempty(index_rows)
    T_index = cell2table(index_rows, 'VariableNames', index_varnames);
    T_index = assign_split_by_file_local(T_index, cfg);
else
    T_index = cell2table(cell(0, length(index_varnames)), ...
        'VariableNames', index_varnames);
end

failure_varnames = {'Mod','Sub','Turbulence','SigIdx','RxFrameIdx', ...
    'Stage','Message','Path'};

if ~isempty(failure_rows)
    T_fail = cell2table(failure_rows, 'VariableNames', failure_varnames);
else
    T_fail = cell2table(cell(0, length(failure_varnames)), ...
        'VariableNames', failure_varnames);
end

file_snr_varnames = {'Mod','Sub','Turbulence','SigIdx','FileKey', ...
    'FileSNR_EVM_dB','MeanFrameSNR_EVM_dB','MedianFrameSNR_EVM_dB', ...
    'StdFrameSNR_EVM_dB','NFrames','FileScMeanSNR_EVM_dB','FileScStdSNR_EVM_dB'};

if ~isempty(file_snr_rows)
    T_file_snr = cell2table(file_snr_rows, 'VariableNames', file_snr_varnames);
else
    T_file_snr = cell2table(cell(0, length(file_snr_varnames)), ...
        'VariableNames', file_snr_varnames);
end

%% ===================== 6. Save index/log/SNR summary =====================

writetable(T_index, fullfile(cfg.out_root, 'dataset_index.csv'));
writetable(T_index, fullfile(cfg.out_root, 'split_index.csv'));
writetable(T_fail, fullfile(dirs.logs, 'failure_log.csv'));
writetable(T_file_snr, fullfile(dirs.snr, 'snr_per_file.csv'));

snr_summary = struct();
snr_summary.frame_snr_evm = all_frame_snr_evm;
snr_summary.file_snr_evm = all_file_snr_evm;
snr_summary.sc_snr_evm = all_sc_snr_evm;
snr_summary.mod_labels = all_mod_labels;
snr_summary.sub_labels = all_sub_labels;
snr_summary.turb_labels = all_turb_labels;
snr_summary.stats = stats;
snr_summary.cfg = cfg;

if ~isempty(all_frame_snr_evm)
    snr_summary.frame_mean = mean(all_frame_snr_evm);
    snr_summary.frame_median = median(all_frame_snr_evm);
    snr_summary.frame_std = std(all_frame_snr_evm);
end

if ~isempty(all_file_snr_evm)
    snr_summary.file_mean = mean(all_file_snr_evm);
    snr_summary.file_median = median(all_file_snr_evm);
    snr_summary.file_std = std(all_file_snr_evm);
end

save(fullfile(dirs.snr, 'snr_summary_evm.mat'), 'snr_summary', '-v7.3');

%% ===================== 7. Print summary =====================

fprintf('\n============================================\n');
fprintf('  Batch Iterative RX1 Dataset Build Complete\n');
fprintf('============================================\n');
fprintf('Total files        : %d\n', stats.total_files);
fprintf('Read OK / Fail     : %d / %d\n', stats.ok_read, stats.fail_read);
fprintf('TX ref OK / Fail   : %d / %d\n', stats.ok_tx_ref, stats.fail_tx_ref);
fprintf('Saved frames       : %d\n', stats.ok_saved);
fprintf('Extract fail       : %d\n', stats.fail_extract);

if ~isempty(all_frame_snr_evm)
    fprintf('\nEVM-SNR Frame mean/median/std: %.2f / %.2f / %.2f dB\n', ...
        mean(all_frame_snr_evm), median(all_frame_snr_evm), std(all_frame_snr_evm));
end

if ~isempty(all_file_snr_evm)
    fprintf('EVM-SNR File  mean/median/std: %.2f / %.2f / %.2f dB\n', ...
        mean(all_file_snr_evm), median(all_file_snr_evm), std(all_file_snr_evm));
end

fprintf('\nOutput root:\n%s\n', cfg.out_root);

%% ===================== 8. Plot SNR summary =====================

if ~isempty(all_frame_snr_evm)
    plot_dataset_snr_summary_local(all_frame_snr_evm, all_file_snr_evm, ...
        all_sc_snr_evm, all_mod_labels, dirs.snr);
end

fprintf('\n===== DONE =====\n');

%% =====================================================================
%% Helper Functions
%% =====================================================================

function make_dir_local(p)
    if ~exist(p, 'dir')
        mkdir(p);
    end
end

function [Mq, bits] = mod_to_order_bits_local(mod_name)
    switch mod_name
        case 'QPSK'
            Mq = 4; bits = 2;
        case '16QAM'
            Mq = 16; bits = 4;
        case '32QAM'
            Mq = 32; bits = 5;
        case '64QAM'
            Mq = 64; bits = 6;
        case '128QAM'
            Mq = 128; bits = 7;
        case '256QAM'
            Mq = 256; bits = 8;
        otherwise
            error('Unknown modulation: %s', mod_name);
    end
end

%% ===================== Robust Keysight .bin reader =====================

function [y, info] = read_keysight_bin_robust_local(filename)
    % 先按标准Keysight头读取；失败后尝试根据剩余字节推断 bpp。
    info = struct();
    info.method = 'unknown';
    info.standard_error = '';
    info.infer_error = '';

    try
        [y, info] = read_keysight_bin_standard_or_infer_local(filename, false);
        info.method = 'standard';
        return;
    catch ME1
        info.standard_error = ME1.message;
    end

    try
        [y, info] = read_keysight_bin_standard_or_infer_local(filename, true);
        info.method = 'inferred_bpp';
        return;
    catch ME2
        info.infer_error = ME2.message;
    end

    error('read_keysight_bin_robust failed: standard=[%s], infer=[%s]', ...
        info.standard_error, info.infer_error);
end

function [y, info] = read_keysight_bin_standard_or_infer_local(filename, force_infer)

    fid = fopen(filename, 'rb', 'ieee-le');
    if fid == -1
        error('Cannot open: %s', filename);
    end
    cleaner = onCleanup(@() fclose(fid));

    fread(fid, 2, '*char')';    % cookie
    fread(fid, 2, '*char')';    % version
    fread(fid, 1, 'int32');     % file_size
    fread(fid, 1, 'int32');     % num_waveforms
    fread(fid, 1, 'int32');     % header_size
    fread(fid, 1, 'int32');     % wave_type
    fread(fid, 1, 'int32');     % num_buffers

    num_points = fread(fid, 1, 'int32');

    fread(fid, 1, 'int32');     % count
    fread(fid, 1, 'float32');   % x_disp_range
    fread(fid, 1, 'float64');   % x_disp_orig
    fread(fid, 1, 'float64');   % x_inc
    fread(fid, 1, 'float64');   % x_orig
    fread(fid, 1, 'int32');     % x_units
    fread(fid, 1, 'int32');     % y_units
    fread(fid, 16, '*char')';   % date
    fread(fid, 16, '*char')';   % time
    fread(fid, 24, '*char')';   % frame
    fread(fid, 16, '*char')';   % wave
    fread(fid, 1, 'float64');   % time tag
    fread(fid, 1, 'uint32');    % segment index
    fread(fid, 1, 'int32');     % data header size
    fread(fid, 1, 'int16');     % buffer type

    bpp_read = fread(fid, 1, 'int16');
    buffer_size = fread(fid, 1, 'int32');

    data_start = ftell(fid);
    d = dir(filename);
    remain_bytes = d.bytes - data_start;

    if isempty(num_points) || numel(num_points) ~= 1 || num_points <= 0
        error('invalid num_points');
    end

    if ~force_infer
        if isempty(bpp_read) || numel(bpp_read) ~= 1 || ...
                ~ismember(double(bpp_read), [1 2 4 8])
            error('invalid bpp: %s', mat2str(bpp_read));
        end
        bpp_candidates = double(bpp_read);
    else
        bpp_candidates = [];

        if ~isempty(bpp_read) && numel(bpp_read) == 1 && ...
                ismember(double(bpp_read), [1 2 4 8])
            bpp_candidates(end+1) = double(bpp_read);
        end

        if ~isempty(buffer_size) && numel(buffer_size) == 1 && buffer_size > 0
            bpp_candidates(end+1) = round(double(buffer_size) / double(num_points));
        end

        if remain_bytes > 0
            bpp_candidates(end+1) = round(double(remain_bytes) / double(num_points));
        end

        bpp_candidates = unique([bpp_candidates 4 2 1 8], 'stable');
        bpp_candidates = bpp_candidates(ismember(bpp_candidates, [1 2 4 8]));
    end

    last_msg = '';

    for ii = 1:length(bpp_candidates)

        bpp = bpp_candidates(ii);

        try
            fseek(fid, data_start, 'bof');
            raw = read_raw_by_bpp_local(fid, double(num_points), bpp);

            if isempty(raw) || length(raw) < 1000
                error('raw too short');
            end

            raw = double(raw(:));

            if std(raw(1:min(5000,end))) == 0
                error('zero variance raw');
            end

            y = raw_to_complex_local(raw);

            info = struct();
            info.method = 'standard_or_infer';
            info.bpp_read = double(bpp_read);
            info.bpp_used = bpp;
            info.num_points = double(num_points);
            info.buffer_size = double(buffer_size);
            info.data_start = data_start;
            info.remain_bytes = remain_bytes;
            return;

        catch ME
            last_msg = ME.message;
        end
    end

    error('all bpp candidates failed: %s', last_msg);
end

function raw = read_raw_by_bpp_local(fid, num_points, bpp)
    switch bpp
        case 4
            raw = fread(fid, num_points, 'float32');
        case 2
            raw = fread(fid, num_points, 'int16');
        case 1
            raw = fread(fid, num_points, 'int8');
        case 8
            raw = fread(fid, num_points, 'double');
        otherwise
            error('unsupported bpp=%g', bpp);
    end
end

function y = raw_to_complex_local(raw)
    raw = double(raw(:));
    n = floor(length(raw) / 2);
    raw = raw(1:2*n);
    y = raw(1:2:end) + 1j * raw(2:2:end);
end

%% ===================== RX1-style sync =====================

function [lts_start, frame_start, info] = find_one_frame_start_rx1_style_local(rx, cfg)

    rx = rx(:).';

    n_fft = cfg.n_fft;
    n_guard = cfg.n_guard;
    sym_len = cfg.sym_len;
    n_syms = cfg.n_syms;
    zeros_head = cfg.zeros_head;

    symbol_bits = zeros_head + n_guard + 2*n_fft + sym_len*n_syms;

    search_len = min(length(rx), 2 * symbol_bits);

    if search_len < symbol_bits
        error('input too short for one-frame sync: search_len=%d', search_len);
    end

    search_sig = rx(1:search_len);

    [detected_packet, edge_index] = packet_edge_power_dect(search_sig, zeros_head);

    load('LongTrainSym_ini.mat', 'LongTrainSym_ini');

    LTS_f = LongTrainSym_ini(1:n_fft);
    LTS_f([1 n_fft/2+1]) = 0;

    ltrs_in = LTS_f;
    ltrs_in(1, n_fft/2+2:n_fft) = conj(ltrs_in(1, n_fft/2:-1:2));

    [fine_time_est, data_df, max_peak_long] = ...
        rx_fine_time_sync_cross_corr( ...
        detected_packet, n_guard, ltrs_in, zeros_head, 0);

    lts_start = edge_index + fine_time_est - 1;
    frame_start = lts_start - (zeros_head + n_guard);

    if frame_start < 1
        frame_start = 1;
    end

    if lts_start < 1 || lts_start + 2*n_fft - 1 > length(rx)
        error('LTS out of range: lts_start=%d, len=%d', lts_start, length(rx));
    end

    info = struct();
    info.edge_index = edge_index;
    info.fine_time_est = fine_time_est;
    info.lts_start = lts_start;
    info.frame_start = frame_start;
    info.data_df = data_df;
    info.max_peak_long = max_peak_long;
    info.symbol_bits = symbol_bits;
end

%% ===================== Direct demodulation from LTS =====================

function [rx_sc, rx_time, rx_frame16_lts, info] = ...
    demod_one_frame_from_lts_start_local(rx, lts_start, cfg)

    rx = rx(:).';

    n_fft = cfg.n_fft;
    n_guard = cfg.n_guard;
    sym_len = cfg.sym_len;
    n_syms = cfg.n_syms;
    carrier_loc = cfg.carrier_loc;

    load('LongTrainSym_ini.mat', 'LongTrainSym_ini');

    LTS_f0 = LongTrainSym_ini(1:n_fft);
    LTS_f0([1 n_fft/2+1]) = 0;

    ltrs_in = LTS_f0;
    ltrs_in(1, n_fft/2+2:n_fft) = conj(ltrs_in(1, n_fft/2:-1:2));
    LTS_f_ref = ltrs_in(:);

    if lts_start < 1 || lts_start + 2*n_fft - 1 > length(rx)
        error('frame too short for LTS: remaining=%d', length(rx) - lts_start + 1);
    end

    % 只对 lts_start 之后的片段做CFO补偿，避免整段长序列相位参考混乱
    seg = rx(lts_start:end);

    if length(seg) < 2*n_fft
        error('frame too short for LTS segment: len=%d', length(seg));
    end

    lts1 = seg(1:n_fft);
    lts2 = seg(n_fft+1:2*n_fft);

    cfo = angle(sum(lts1(:).*conj(lts2(:)))) / (2*pi*n_fft);

    n = 0:length(seg)-1;
    seg_cfo = seg .* exp(-1j*2*pi*cfo*n/n_fft);

    lts1c = seg_cfo(1:n_fft);
    lts2c = seg_cfo(n_fft+1:2*n_fft);

    data_start = 2*n_fft + 1;
    dp_all = seg_cfo(data_start:end);

    nd = floor(length(dp_all) / sym_len);

    if nd <= 0
        error('no complete OFDM symbols: nd=0');
    end

    n_use = min(nd, n_syms);

    if n_use < cfg.min_ofdm_symbols
        error('too few OFDM symbols: n_use=%d', n_use);
    end

    dp = dp_all(1:n_use * sym_len);
    dm = reshape(dp, sym_len, n_use);
    dn = dm(n_guard+1:end, :);

    fd = fft(dn, n_fft, 1) ./ sqrt(n_fft);

    lts_avg = (lts1c(:) + lts2c(:)) ./ 2;
    lts_fd = fft(lts_avg, n_fft) ./ sqrt(n_fft);

    H = lts_fd ./ (LTS_f_ref(:) + 1e-12);
    H(abs(LTS_f_ref(:)) < 0.5) = 1;

    feq = fd ./ H;
    rx_sc = feq(carrier_loc, :);

    % 时域输入：取数据区前 M_time 点，不足补零
    rx_payload_time = dp_all(:).';

    if length(rx_payload_time) >= cfg.M_time
        rx_time = rx_payload_time(1:cfg.M_time);
    else
        rx_time = [rx_payload_time, zeros(1, cfg.M_time - length(rx_payload_time))];
    end

    % 保存 LTS 对齐后帧片段
    n_save = min(length(seg_cfo), 2*n_fft + n_syms * sym_len);
    rx_frame16_lts = seg_cfo(1:n_save);

    info = struct();
    info.cfo = cfo;
    info.n_use = n_use;
    info.nd_available = nd;
    info.lts_start = lts_start;
    info.data_start_rel = data_start;
    info.data_end_rel = data_start + n_use*sym_len - 1;
end

%% ===================== EVM-SNR matching =====================

function [best_tx_id, best_snr_db, best_snr_sc_db, snr_list_db, margin_db] = ...
    match_rx_to_tx_by_evm_snr_local(rx_sc, tx_refs, cfg)

    n_ref = length(tx_refs);

    snr_list_db = NaN(1, n_ref);
    snr_sc_all = cell(1, n_ref);

    for tid = 1:n_ref

        tx_ref = tx_refs{tid};

        n_sc = min([size(rx_sc,1), size(tx_ref,1), cfg.n_sc]);
        n_sym = min([size(rx_sc,2), size(tx_ref,2), cfg.n_syms]);

        rx_use = rx_sc(1:n_sc, 1:n_sym);
        tx_use = tx_ref(1:n_sc, 1:n_sym);

        snr_sc = NaN(n_sc, 1);

        for sc = 1:n_sc

            r = rx_use(sc, :);
            t = tx_use(sc, :);

            good = isfinite(real(r)) & isfinite(imag(r)) & ...
                   isfinite(real(t)) & isfinite(imag(t));

            r = r(good);
            t = t(good);

            if length(r) < 10
                continue;
            end

            % 最小二乘复增益校正：r ≈ a*t + e
            a = (r * t') / (t * t' + eps);
            e = r - a * t;

            sig_pow = mean(abs(a * t).^2);
            noise_pow = mean(abs(e).^2);

            snr_sc(sc) = 10 * log10(sig_pow / (noise_pow + eps) + eps);
        end

        valid = snr_sc(isfinite(snr_sc));

        if ~isempty(valid)
            snr_list_db(tid) = 10 * log10(mean(10.^(valid / 10)));
        end

        snr_sc_all{tid} = snr_sc;
    end

    [best_snr_db, best_tx_id] = max(snr_list_db);

    if isempty(best_snr_db) || ~isfinite(best_snr_db)
        best_tx_id = NaN;
        best_snr_db = NaN;
        best_snr_sc_db = NaN(cfg.n_sc, 1);
        margin_db = NaN;
    else
        best_snr_sc_db = snr_sc_all{best_tx_id};

        sorted_snr = sort(snr_list_db(isfinite(snr_list_db)), 'descend');

        if length(sorted_snr) >= 2
            margin_db = sorted_snr(1) - sorted_snr(2);
        else
            margin_db = NaN;
        end
    end
end

%% ===================== CDM =====================

function cdm = make_cdm_from_rxsc_local(rx_sc, nbin, clip_val)

    z = rx_sc(:);
    z = z(isfinite(real(z)) & isfinite(imag(z)));

    if isempty(z)
        cdm = zeros(nbin, nbin);
        return;
    end

    z = z - mean(z);
    z = z ./ (rms(abs(z)) + eps);

    zr = max(min(real(z), clip_val), -clip_val);
    zi = max(min(imag(z), clip_val), -clip_val);

    edges = linspace(-clip_val, clip_val, nbin + 1);

    H = histcounts2(zi, zr, edges, edges);

    cdm = log1p(H);
    cdm = cdm ./ (max(cdm(:)) + eps);
end

%% ===================== Split by file_key =====================

function T = assign_split_by_file_local(T, cfg)

    rng(cfg.random_seed);

    n = height(T);
    T.Split = repmat({'unused'}, n, 1);

    if n == 0
        return;
    end

    % 每个 Mod × Sub 内按 FileKey 划分，确保同一文件3帧不跨集合
    group_key = strcat(T.Mod, {'|'}, T.Sub);
    groups = unique(group_key);

    for gi = 1:length(groups)

        g = groups{gi};
        mask_g = strcmp(group_key, g);

        keys = unique(T.FileKey(mask_g));
        keys = keys(randperm(length(keys)));

        nk = length(keys);

        n_train = floor(cfg.train_ratio * nk);
        n_val = floor(cfg.val_ratio * nk);

        train_keys = keys(1:n_train);
        val_keys = keys(n_train+1 : min(n_train+n_val, nk));
        test_keys = keys(min(n_train+n_val, nk)+1 : end);

        for i = 1:length(train_keys)
            T.Split(strcmp(T.FileKey, train_keys{i})) = {'train'};
        end

        for i = 1:length(val_keys)
            T.Split(strcmp(T.FileKey, val_keys{i})) = {'val'};
        end

        for i = 1:length(test_keys)
            T.Split(strcmp(T.FileKey, test_keys{i})) = {'test'};
        end
    end
end

%% ===================== Plot summary =====================

function plot_dataset_snr_summary_local(all_frame_snr, all_file_snr, ...
    all_sc_snr, all_mod_labels, out_dir)

    figure('Position', [100 100 1400 900]);

    subplot(2, 3, 1);
    histogram(all_frame_snr, 50, 'FaceColor', '#0072BD', 'EdgeColor', 'none');
    xlabel('Frame EVM-SNR (dB)');
    ylabel('Count');
    title(sprintf('Frame EVM-SNR (n=%d, mean=%.2f dB)', ...
        length(all_frame_snr), mean(all_frame_snr)));
    grid on;

    subplot(2, 3, 2);
    hold on;

    mods = unique(all_mod_labels);
    colors = lines(length(mods));

    for i = 1:length(mods)
        mask = strcmp(all_mod_labels, mods{i});
        histogram(all_frame_snr(mask), 30, ...
            'FaceColor', colors(i,:), ...
            'EdgeColor', 'none', ...
            'FaceAlpha', 0.45);
    end

    legend(mods, 'Location', 'northwest');
    xlabel('Frame EVM-SNR (dB)');
    ylabel('Count');
    title('Frame EVM-SNR by Modulation');
    grid on;

    subplot(2, 3, 3);
    histogram(all_file_snr, 40, 'FaceColor', '#D95319', 'EdgeColor', 'none');
    xlabel('File EVM-SNR (dB)');
    ylabel('Count');
    title(sprintf('File EVM-SNR (n=%d, mean=%.2f dB)', ...
        length(all_file_snr), mean(all_file_snr)));
    grid on;

    subplot(2, 3, 4);

    if ~isempty(all_sc_snr)
        imagesc(all_sc_snr);
        colorbar;
        xlabel('Frame Index');
        ylabel('Subcarrier Index');
        title(sprintf('Subcarrier EVM-SNR (%d frames)', size(all_sc_snr, 2)));
        colormap('jet');
        set(gca, 'YDir', 'normal');
    end

    subplot(2, 3, 5);
    [f, x] = ecdf(all_frame_snr);
    plot(x, f*100, 'LineWidth', 2, 'Color', '#0072BD');
    xlabel('Frame EVM-SNR (dB)');
    ylabel('Cumulative (%)');
    title('Cumulative Frame EVM-SNR');
    grid on;
    ylim([0 100]);

    subplot(2, 3, 6);
    boxchart(categorical(all_mod_labels), all_frame_snr);
    xlabel('Modulation');
    ylabel('Frame EVM-SNR (dB)');
    title('Frame EVM-SNR Boxplot by Modulation');
    grid on;

    saveas(gcf, fullfile(out_dir, 'dataset_snr_summary_evm.png'));
end
