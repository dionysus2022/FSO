%% build_dataset_after_snr_v1.m
% 正式数据集生成脚本
% 基于当前已验证的 batch_process_all_samples.m 流程
%
% 功能：
% 1. 遍历全部 mod/sub/sig_idx
% 2. 修复/兼容 32QAM/sub3 的 bpp 异常读取问题
% 3. AWG循环同步
% 4. 三帧提取 + LTS解调 + CFO补偿 + FFT均衡
% 5. 与 tx_frame1/2/3 自动匹配，计算 EVM-SNR
% 6. 保存 time_32768 / freq_sc / cdm_64 三类特征
% 7. 输出 dataset_index.csv、failure_log.csv、snr_per_file.csv
% 8. 按 file_id 划分 train/val/test，避免数据泄漏

clear; clear global; close all; clc;

addpath(genpath('D:\Project_code\OptDSP_FSO_2\practical waterfiling-exp\OptDSP_lite-master'));

global PROG;
PROG.showMessagesLevel = 0;
initProg();

%% ===================== Config =====================

cfg = struct();

cfg.data_root = 'D:\Project_code\OptDSP_FSO_2\practical waterfiling-exp\OptDSP_lite-master\2_Data_Results';
cfg.rx_date   = '2026.06.26';
cfg.tx_root   = fullfile(cfg.data_root, 'tx_3frame_6mod');
cfg.out_root  = fullfile(cfg.data_root, 'dataset_final_v1');

cfg.Fs_rx   = 80e9;
cfg.Fs_base = 16e9;

cfg.n_frames = 3;
cfg.M_time = 32768;

cfg.cdm_bins = 64;
cfg.cdm_clip = 3.0;

cfg.mod_names = {'QPSK','16QAM','32QAM','64QAM','128QAM','256QAM'};
cfg.sub_list  = {'sub1','sub2','sub3'};

cfg.turb_map = containers.Map('KeyType','char','ValueType','char');
cfg.turb_map('sub1') = 'weak';
cfg.turb_map('sub2') = 'moderate';
cfg.turb_map('sub3') = 'strong';

% OFDM params
cfg.zeros_head = 80;
cfg.n_fft = 256;
cfg.n_guard = 16;
cfg.n_syms = 128;
cfg.carrier_loc = 4:126;
cfg.n_sc = length(cfg.carrier_loc);
cfg.sym_len = cfg.n_fft + cfg.n_guard;
cfg.header_len_16 = cfg.zeros_head + cfg.n_guard + 2*cfg.n_fft;
cfg.frame_len_16 = cfg.header_len_16 + cfg.sym_len * cfg.n_syms;

% Sync params，沿用当前正确版本
cfg.sync_decim = 20;
cfg.num_corr_candidates = 60;
cfg.frame_margin_80 = 8000;
cfg.fine_search_len_80 = 30000;

% Valid criteria
cfg.min_ofdm_symbols = 100;
cfg.min_snr_evm_db = -5;
cfg.min_snr_sc_valid_ratio = 0.80;

% Split
cfg.random_seed = 2026;
cfg.train_ratio = 0.70;
cfg.val_ratio = 0.15;
cfg.test_ratio = 0.15;

%% ===================== Output dirs =====================

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

LTS = make_lts_local(cfg.n_fft);

%% ===================== Accumulators =====================

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
stats.ok_tx_txt = 0;
stats.fail_tx_txt = 0;
stats.ok_tx_ref = 0;
stats.fail_tx_ref = 0;
stats.ok_sync = 0;
stats.fail_sync = 0;
stats.ok_demod = 0;
stats.fail_demod = 0;
stats.ok_saved = 0;

%% ===================== Main Loop =====================

for mi = 1:length(cfg.mod_names)

    mod_name = cfg.mod_names{mi};
    label_id = mi - 1;

    [Mq, bits] = mod_to_order_bits_local(mod_name);
    nBpS_net = bits - 0.2 * (bits > 2);

    TX.SIG = setSignalParams('symRate', 8e9, 'M', Mq, ...
        'nPol', 1, 'nBpS', nBpS_net, 'nSyms', cfg.n_syms, ...
        'roll-off', 0.25, 'modulation', 'QAM');

    TX.QAM = QAM_config(TX.SIG);

    rx_mod_dir = fullfile(cfg.data_root, 'rx_data', cfg.rx_date, mod_name);

    if ~exist(rx_mod_dir, 'dir')
        fprintf('[Skip] missing rx mod dir: %s\n', rx_mod_dir);
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
            fprintf('[Skip] missing rx sub dir: %s\n', rx_dir);
            continue;
        end

        bin_list = dir(fullfile(rx_dir, '*.bin'));

        if isempty(bin_list)
            fprintf('[Skip] no bin files: %s\n', rx_dir);
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
            tx_txt = fullfile(cfg.tx_root, mod_name, sub_name, ...
                sprintf('sig_%04d.txt', sig_idx));

            file_key = sprintf('%s_%s_sig%04d', mod_name, sub_name, sig_idx);

            %% ---------- A. Read RX bin ----------

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

            %% ---------- B. Load TX txt ----------

            if ~exist(tx_txt, 'file')

                stats.fail_tx_txt = stats.fail_tx_txt + 1;

                failure_rows(end+1,:) = {mod_name, sub_name, turb_name, ...
                    sig_idx, 0, 'tx_txt_missing', 'missing tx txt', tx_txt};

                continue;
            end

            try
                tx_ref80 = load_ascii_complex_local(tx_txt);
                tx_ref80 = tx_ref80(:);

                frame_len_80 = floor(length(tx_ref80) / cfg.n_frames);

                if frame_len_80 <= 0
                    error('invalid frame_len_80');
                end

                stats.ok_tx_txt = stats.ok_tx_txt + 1;

            catch ME
                stats.fail_tx_txt = stats.fail_tx_txt + 1;

                failure_rows(end+1,:) = {mod_name, sub_name, turb_name, ...
                    sig_idx, 0, 'tx_txt_load', ME.message, tx_txt};

                continue;
            end

            %% ---------- C. Load TX frame refs ----------

            tx_refs = cell(1, cfg.n_frames);
            ref_ok = true;
            ref_msg = '';

            for tid = 1:cfg.n_frames

                ref_file = fullfile(cfg.tx_root, mod_name, sub_name, ...
                    sprintf('sig_%04d_frame%d.mat', sig_idx, tid));

                if ~exist(ref_file, 'file')
                    ref_ok = false;
                    ref_msg = ['missing: ' ref_file];
                    break;
                end

                tmp_ref = load(ref_file);

                if ~isfield(tmp_ref, 'data_tx')
                    ref_ok = false;
                    ref_msg = ['no data_tx: ' ref_file];
                    break;
                end

                tx_refs{tid} = tmp_ref.data_tx.';   % [123 × 128]
            end

            if ~ref_ok
                stats.fail_tx_ref = stats.fail_tx_ref + 1;

                failure_rows(end+1,:) = {mod_name, sub_name, turb_name, ...
                    sig_idx, 0, 'tx_ref', ref_msg, ''};

                continue;
            else
                stats.ok_tx_ref = stats.ok_tx_ref + 1;
            end

            %% ---------- D. AWG cycle sync ----------

            try
                [rx_start_80, start_tx_id, sync_metric] = ...
                    find_awg_cycle_start_local(rx80, tx_ref80, frame_len_80, cfg);

                stats.ok_sync = stats.ok_sync + 1;

            catch ME
                stats.fail_sync = stats.fail_sync + 1;

                failure_rows(end+1,:) = {mod_name, sub_name, turb_name, ...
                    sig_idx, 0, 'sync', ME.message, rx_bin};

                continue;
            end

            %% ---------- E. Process 3 frames ----------

            file_frame_snr_evm = [];
            file_sc_snr_mat = [];

            for rk = 1:cfg.n_frames

                expected_tx_id = mod(start_tx_id + rk - 2, cfg.n_frames) + 1;
                seg_start_nom = rx_start_80 + (rk - 1) * frame_len_80;

                try
                    [rx_raw80, seg_start80, seg_end80] = ...
                        extract_frame_with_margin_local(rx80, seg_start_nom, ...
                        frame_len_80, cfg.frame_margin_80);

                    rx_raw16 = resample(rx_raw80(:).', cfg.Fs_base, cfg.Fs_rx);
                    rx_raw16 = rx_raw16 - mean(rx_raw16);
                    rx_raw16 = rx_raw16 ./ (rms(rx_raw16) + eps);

                    [rx_sc, rx_time, rx_frame16_lts, demod_info] = ...
                        demod_frame_from_window_local(rx_raw16, LTS, cfg);

                    [best_tx_id, snr_frame_evm_db, snr_sc_evm_db, ...
                        snr_list_db, snr_match_margin_db] = ...
                        match_rx_to_tx_by_evm_snr_local(rx_sc, tx_refs, cfg);

                    valid_snr_sc = snr_sc_evm_db(isfinite(snr_sc_evm_db));

                    snr_sc_valid_ratio = length(valid_snr_sc) / length(snr_sc_evm_db);

                    if isempty(valid_snr_sc)
                        snr_sc_mean_db = NaN;
                        snr_sc_std_db = NaN;
                        snr_sc_min_db = NaN;
                        snr_sc_max_db = NaN;
                    else
                        snr_sc_mean_db = mean(valid_snr_sc);
                        snr_sc_std_db = std(valid_snr_sc);
                        snr_sc_min_db = min(valid_snr_sc);
                        snr_sc_max_db = max(valid_snr_sc);
                    end

                    snr_power_frame_db = 10 * log10(mean(abs(rx_sc(:)).^2) + eps);

                    valid_clean = ...
                        demod_info.n_use >= cfg.min_ofdm_symbols && ...
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

                    %% ---------- Save samples ----------

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
                    sample.rx_frame_16_lts_aligned = single(rx_frame16_lts);
                    sample.rx_time = single(rx_time);
                    sample.rx_sc = single(rx_sc);
                    sample.cdm64 = single(cdm64);
                    sample.snr_sc_evm_db = single(snr_sc_evm_db);
                    sample.snr_frame_evm_db = snr_frame_evm_db;
                    sample.snr_power_frame_db = snr_power_frame_db;
                    sample.best_tx_frame_id = best_tx_id;
                    sample.expected_tx_frame_id = expected_tx_id;
                    sample.snr_match_margin_db = snr_match_margin_db;
                    sample.snr_list_db = snr_list_db;
                    sample.mod_name = mod_name;
                    sample.label_id = label_id;
                    sample.mod_order = Mq;
                    sample.sub_name = sub_name;
                    sample.turbulence = turb_name;
                    sample.sig_idx = sig_idx;
                    sample.rx_frame_idx = rk;
                    sample.file_key = file_key;
                    sample.rx_bin = rx_bin;
                    sample.tx_txt = tx_txt;
                    sample.seg_start80 = seg_start80;
                    sample.seg_end80 = seg_end80;
                    sample.rx_start_80 = rx_start_80;
                    sample.sync_metric = sync_metric;
                    sample.start_tx_id = start_tx_id;
                    sample.read_info = read_info;
                    sample.demod_info = demod_info;
                    sample.valid_flag = valid_flag;
                    sample.valid_reason = valid_reason;

                    save(full_path, 'sample', '-v7.3');

                    sample_time = struct();
                    sample_time.rx_time = single(rx_time);
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

                    %% ---------- Index row ----------

                    index_rows(end+1,:) = { ...
                        full_path, time_path, freq_path, cdm_path, ...
                        file_key, mod_name, label_id, Mq, sub_name, turb_name, ...
                        sig_idx, rk, expected_tx_id, best_tx_id, ...
                        snr_frame_evm_db, snr_power_frame_db, ...
                        snr_sc_mean_db, snr_sc_std_db, snr_sc_min_db, snr_sc_max_db, ...
                        snr_sc_valid_ratio, snr_match_margin_db, ...
                        demod_info.n_use, demod_info.cfo, ...
                        rx_start_80, seg_start80, seg_end80, sync_metric, ...
                        valid_flag, valid_reason};

                    stats.ok_demod = stats.ok_demod + 1;
                    stats.ok_saved = stats.ok_saved + 1;

                    file_frame_snr_evm(end+1) = snr_frame_evm_db;
                    file_sc_snr_mat = [file_sc_snr_mat, snr_sc_evm_db(:)];

                    all_frame_snr_evm(end+1) = snr_frame_evm_db;
                    all_sc_snr_evm = [all_sc_snr_evm, snr_sc_evm_db(:)];
                    all_mod_labels{end+1} = mod_name;
                    all_sub_labels{end+1} = sub_name;
                    all_turb_labels{end+1} = turb_name;

                catch ME
                    stats.fail_demod = stats.fail_demod + 1;

                    failure_rows(end+1,:) = {mod_name, sub_name, turb_name, ...
                        sig_idx, rk, 'extract_demod_save', ME.message, rx_bin};
                end
            end

            %% ---------- File-level SNR ----------

            if ~isempty(file_frame_snr_evm)

                file_snr_evm_db = 10 * log10(mean(10.^(file_frame_snr_evm / 10)));
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
                    mean(file_frame_snr_evm), median(file_frame_snr_evm), ...
                    std(file_frame_snr_evm), length(file_frame_snr_evm), ...
                    file_sc_mean_db, file_sc_std_db};
            end

            if mod(bi, 15) == 0 || bi == length(bin_list)
                fprintf('%s/%s: %d/%d files | read=%d | demod=%d | fail_demod=%d\n', ...
                    mod_name, sub_name, bi, length(bin_list), ...
                    stats.ok_read, stats.ok_demod, stats.fail_demod);
            end
        end
    end
end

%% ===================== Build tables =====================

index_varnames = {'FullFramePath','TimePath','FreqPath','CDMPath', ...
    'FileKey','Mod','LabelID','ModOrder','Sub','Turbulence', ...
    'SigIdx','RxFrameIdx','ExpectedTxFrameID','BestTxFrameID', ...
    'SNRFrameEVM_dB','SNRFramePower_dB', ...
    'SNRScMean_dB','SNRScStd_dB','SNRScMin_dB','SNRScMax_dB', ...
    'SNRScValidRatio','SNRMatchMargin_dB', ...
    'NOFDMSymbols','CFO', ...
    'RxStart80','SegStart80','SegEnd80','SyncMetric', ...
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

%% ===================== Save outputs =====================

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
    snr_summary.file_mean = mean(all_file_snr_evm);
    snr_summary.file_median = median(all_file_snr_evm);
    snr_summary.file_std = std(all_file_snr_evm);
end

save(fullfile(dirs.snr, 'snr_summary_evm.mat'), 'snr_summary', '-v7.3');

%% ===================== Print summary =====================

fprintf('\n============================================\n');
fprintf('  Final Dataset Build Complete\n');
fprintf('============================================\n');
fprintf('Total files        : %d\n', stats.total_files);
fprintf('Read OK / Fail     : %d / %d\n', stats.ok_read, stats.fail_read);
fprintf('TX txt OK / Fail   : %d / %d\n', stats.ok_tx_txt, stats.fail_tx_txt);
fprintf('TX ref OK / Fail   : %d / %d\n', stats.ok_tx_ref, stats.fail_tx_ref);
fprintf('Sync OK / Fail     : %d / %d\n', stats.ok_sync, stats.fail_sync);
fprintf('Demod OK / Fail    : %d / %d\n', stats.ok_demod, stats.fail_demod);
fprintf('Saved frames       : %d\n', stats.ok_saved);

if ~isempty(all_frame_snr_evm)
    fprintf('\nEVM-SNR Frame mean/median/std: %.2f / %.2f / %.2f dB\n', ...
        mean(all_frame_snr_evm), median(all_frame_snr_evm), std(all_frame_snr_evm));
    fprintf('EVM-SNR File mean/median/std : %.2f / %.2f / %.2f dB\n', ...
        mean(all_file_snr_evm), median(all_file_snr_evm), std(all_file_snr_evm));
end

fprintf('\nOutput root:\n%s\n', cfg.out_root);

%% ===================== Plot summary =====================

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
            error('unknown mod: %s', mod_name);
    end
end

function x = load_ascii_complex_local(filename)

    tmp = load(filename);

    if size(tmp, 2) >= 2
        x = complex(tmp(:,1), tmp(:,2));
    else
        x = tmp(:);
    end
end

function LTS = make_lts_local(n_fft)

    load('LongTrainSym_ini.mat', 'LongTrainSym_ini');

    LTS_f0 = LongTrainSym_ini(1:n_fft);
    LTS_f0([1 n_fft/2+1]) = 0;

    ltrs_in = LTS_f0;
    ltrs_in(1, n_fft/2+2:n_fft) = conj(ltrs_in(1, n_fft/2:-1:2));

    LTS.freq = ltrs_in(:);
    LTS.time = ifft(ltrs_in(:));
end

%% ===================== Robust Keysight Reader =====================

function [y, info] = read_keysight_bin_robust_local(filename)

    info = struct();
    info.method = 'standard';
    info.bpp = NaN;
    info.inferred_bpp = NaN;
    info.num_points = NaN;
    info.data_start = NaN;

    try
        [y, info] = read_keysight_bin_standard_or_infer_local(filename, false);
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

    fread(fid, 2, '*char')';
    fread(fid, 2, '*char')';
    fread(fid, 1, 'int32');
    fread(fid, 1, 'int32');
    fread(fid, 1, 'int32');
    fread(fid, 1, 'int32');
    fread(fid, 1, 'int32');

    num_points = fread(fid, 1, 'int32');

    fread(fid, 1, 'int32');
    fread(fid, 1, 'float32');
    fread(fid, 1, 'float64');
    fread(fid, 1, 'float64');
    fread(fid, 1, 'float64');
    fread(fid, 1, 'int32');
    fread(fid, 1, 'int32');
    fread(fid, 16, '*char')';
    fread(fid, 16, '*char')';
    fread(fid, 24, '*char')';
    fread(fid, 16, '*char')';
    fread(fid, 1, 'float64');
    fread(fid, 1, 'uint32');
    fread(fid, 1, 'int32');
    fread(fid, 1, 'int16');

    bpp_read = fread(fid, 1, 'int16');
    buffer_size = fread(fid, 1, 'int32');

    data_start = ftell(fid);

    d = dir(filename);
    remain_bytes = d.bytes - data_start;

    if isempty(num_points) || numel(num_points) ~= 1 || num_points <= 0
        error('invalid num_points');
    end

    bpp_candidates = [];

    if ~force_infer
        if isempty(bpp_read) || numel(bpp_read) ~= 1 || ...
                ~ismember(double(bpp_read), [1 2 4 8])
            error('invalid bpp: %s', mat2str(bpp_read));
        end

        bpp_candidates = double(bpp_read);
    else
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

    for i = 1:length(bpp_candidates)

        bpp = bpp_candidates(i);

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
            info.bpp = double(bpp_read);
            info.inferred_bpp = bpp;
            info.num_points = double(num_points);
            info.buffer_size = double(buffer_size);
            info.data_start = data_start;

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

%% ===================== Sync Functions =====================

function [start_80, start_tx_id, metric_best] = ...
    find_awg_cycle_start_local(rx80, tx_ref80, frame_len_80, cfg)

    rx80 = rx80(:);
    tx_ref80 = tx_ref80(:);

    if length(tx_ref80) < cfg.n_frames * frame_len_80
        error('tx_ref80 too short');
    end

    if length(rx80) < cfg.n_frames * frame_len_80
        error('rx80 too short');
    end

    decim = cfg.sync_decim;

    rx_env = abs(rx80(1:decim:end));
    rx_env = rx_env - mean(rx_env);
    rx_env = rx_env ./ (std(rx_env) + eps);

    all_score = [];
    all_idx = [];
    all_tid = [];

    for tid = 1:cfg.n_frames

        tx_frame = tx_ref80((tid - 1) * frame_len_80 + (1:frame_len_80));

        tx_env = abs(tx_frame(1:decim:end));
        tx_env = tx_env - mean(tx_env);
        tx_env = tx_env ./ (std(tx_env) + eps);

        if length(rx_env) < length(tx_env)
            continue;
        end

        c = conv(rx_env, flipud(tx_env), 'valid');
        c_abs = abs(c);

        n_take = min(cfg.num_corr_candidates, length(c_abs));
        [vals, idxs] = maxk(c_abs, n_take);

        all_score = [all_score; vals(:)];
        all_idx = [all_idx; idxs(:)];
        all_tid = [all_tid; tid * ones(n_take,1)];
    end

    if isempty(all_score)
        error('no correlation candidate');
    end

    [~, order] = sort(all_score, 'descend');

    for ii = 1:length(order)

        idx_d = all_idx(order(ii));
        tid = all_tid(order(ii));

        coarse_start = (idx_d - 1) * decim + 1;

        start_candidate = refine_start_fullrate_env_local( ...
            rx80, tx_ref80, frame_len_80, tid, coarse_start, cfg);

        tx_id_candidate = tid;

        while start_candidate + cfg.n_frames * frame_len_80 - 1 > length(rx80)
            start_candidate = start_candidate - frame_len_80;
            tx_id_candidate = mod(tx_id_candidate - 2, cfg.n_frames) + 1;
        end

        while start_candidate < 1
            start_candidate = start_candidate + frame_len_80;
            tx_id_candidate = mod(tx_id_candidate, cfg.n_frames) + 1;
        end

        if start_candidate >= 1 && ...
                start_candidate + cfg.n_frames * frame_len_80 - 1 <= length(rx80)

            start_80 = round(start_candidate);
            start_tx_id = tx_id_candidate;
            metric_best = all_score(order(ii));
            return;
        end
    end

    error('cannot contain 3 complete frames');
end

function start_refined = refine_start_fullrate_env_local( ...
    rx80, tx_ref80, frame_len_80, tx_id, coarse_start, cfg)

    tpl = tx_ref80((tx_id - 1) * frame_len_80 + (1:frame_len_80));
    L = min([length(tpl), cfg.fine_search_len_80, length(rx80)]);

    tpl_env = abs(tpl(1:L));
    tpl_env = tpl_env - mean(tpl_env);
    tpl_env = tpl_env ./ (std(tpl_env) + eps);

    win = 4 * cfg.sync_decim;

    s1 = max(1, coarse_start - win);
    s2 = min(length(rx80) - L + 1, coarse_start + win);

    if s2 < s1
        start_refined = coarse_start;
        return;
    end

    best_val = -inf;
    best_s = coarse_start;

    for s = s1:s2

        r_env = abs(rx80(s:s+L-1));
        r_env = r_env - mean(r_env);
        r_env = r_env ./ (std(r_env) + eps);

        val = abs(r_env(:)' * tpl_env(:));

        if val > best_val
            best_val = val;
            best_s = s;
        end
    end

    start_refined = best_s;
end

function [rx_raw80, seg_start, seg_end] = ...
    extract_frame_with_margin_local(rx80, seg_start_nom, frame_len_80, margin)

    seg_start = max(1, seg_start_nom - margin);
    seg_end = min(length(rx80), seg_start_nom + frame_len_80 - 1 + margin);

    if seg_end <= seg_start
        error('invalid segment range');
    end

    rx_raw80 = rx80(seg_start:seg_end);
end

%% ===================== Demod Function =====================

function [rx_sc, rx_time, rx_frame16_lts, info] = ...
    demod_frame_from_window_local(rx_raw16, LTS, cfg)

    rx = rx_raw16(:).';

    n_fft = cfg.n_fft;
    n_guard = cfg.n_guard;
    sym_len = cfg.sym_len;

    LTS_t = LTS.time(:).';

    if length(rx) < 2*n_fft + cfg.min_ofdm_symbols*sym_len
        error('window too short before LTS sync: len=%d', length(rx));
    end

    xc = abs(conv(rx, flipud(conj(LTS_t)), 'valid'));
    [peak_val, lts_peak] = max(xc);

    remaining = length(rx) - lts_peak + 1;

    if remaining < 2*n_fft + cfg.min_ofdm_symbols*sym_len
        error('frame too short after LTS: remaining=%d', remaining);
    end

    lts1 = rx(lts_peak : lts_peak + n_fft - 1);
    lts2 = rx(lts_peak + n_fft : lts_peak + 2*n_fft - 1);

    cfo = angle(sum(lts1(:).*conj(lts2(:)))) / (2*pi*n_fft);

    n = 0:remaining-1;
    rx_comp = rx(lts_peak:end) .* exp(-1j*2*pi*cfo*n/n_fft);

    lts1c = rx_comp(1:n_fft);
    lts2c = rx_comp(n_fft+1:2*n_fft);

    data_start = 2*n_fft + 1;
    dp_all = rx_comp(data_start:end);

    nd = floor(length(dp_all) / sym_len);

    if nd < cfg.min_ofdm_symbols
        error('too few OFDM symbols: nd=%d', nd);
    end

    n_use = min(nd, cfg.n_syms);

    dp = dp_all(1:n_use * sym_len);
    dm = reshape(dp, sym_len, n_use);
    dn = dm(n_guard+1:end, :);

    fd = fft(dn, n_fft, 1) / sqrt(n_fft);

    lts_avg = (lts1c(:) + lts2c(:)) / 2;
    lts_fd = fft(lts_avg, n_fft) / sqrt(n_fft);

    H = lts_fd ./ (LTS.freq(:) + 1e-12);
    H(abs(LTS.freq(:)) < 0.5) = 1;

    feq = fd ./ H;

    rx_sc = feq(cfg.carrier_loc, :);

    rx_payload_time = dp_all(:).';

    if length(rx_payload_time) >= cfg.M_time
        rx_time = rx_payload_time(1:cfg.M_time);
    else
        rx_time = [rx_payload_time, zeros(1, cfg.M_time - length(rx_payload_time))];
    end

    n_save = min(length(rx_comp), 2*n_fft + cfg.n_syms * sym_len);
    rx_frame16_lts = rx_comp(1:n_save);

    info = struct();
    info.lts_peak = lts_peak;
    info.lts_peak_val = peak_val;
    info.cfo = cfo;
    info.n_use = n_use;
    info.nd_available = nd;
    info.data_start = data_start;
    info.remaining = remaining;
end

%% ===================== EVM-SNR Matching =====================

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

            % r ≈ a * t + e
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

%% ===================== Split =====================

function T = assign_split_by_file_local(T, cfg)

    rng(cfg.random_seed);

    n = height(T);
    T.Split = repmat({'unused'}, n, 1);

    if n == 0
        return;
    end

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

%% ===================== Plot =====================

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