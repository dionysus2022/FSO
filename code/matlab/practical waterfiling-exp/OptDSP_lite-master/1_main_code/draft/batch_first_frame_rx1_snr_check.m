%% batch_first_frame_rx1_snr_check.m
% =========================================================
% 目的：
%   每个接收 .bin 只采用 rx1/deOFDM 识别出的第一帧信号；
%   然后与发送端 tx_frame1/2/3.mat 分别匹配；
%   使用 rx1-style: symDemapper + EVM_eval 估计 SNR。
%
% 特点：
%   1. 不再连续切 3 帧；
%   2. 不使用手动 frame extraction；
%   3. 每个 .bin 只输出 1 个样本；
%   4. 用于验证：当前解调流程是否真的可靠。
%
% 注意：
%   实验平台是 IM/DD：MZM + APD + oscilloscope，
%   所以 Keysight .bin 按实值单通道读取，不做 I+jQ 奇偶点组合。
% =========================================================

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
cfg.out_root  = fullfile(cfg.data_root, 'dataset_first_frame_rx1_snr_check');

cfg.Fs_rx   = 80e9;
cfg.Fs_base = 16e9;

cfg.n_frames_tx = 3;

cfg.mod_names = {'QPSK','16QAM','32QAM','64QAM','128QAM','256QAM'};
cfg.sub_list  = {'sub1','sub2','sub3'};

cfg.turb_map = containers.Map('KeyType','char','ValueType','char');
cfg.turb_map('sub1') = 'weak';
cfg.turb_map('sub2') = 'moderate';
cfg.turb_map('sub3') = 'strong';

% OFDM 参数，保持和 rx1 / tx3 一致
cfg.zeros_head = 80;
cfg.n_fft      = 256;
cfg.n_guard    = 16;
cfg.n_syms     = 128;

cfg.carrier_loc = 4:126;
cfg.carrier_loc_demo = [4:126,132:254];
cfg.n_sc = length(cfg.carrier_loc);
cfg.n_sc_demo = length(cfg.carrier_loc_demo);

cfg.sym_len = cfg.n_fft + cfg.n_guard;
cfg.frame_len_16 = cfg.zeros_head + cfg.n_guard + 2*cfg.n_fft + cfg.sym_len*cfg.n_syms;

cfg.M_time = 32768;

% 匹配 tx_frame1/2/3 时允许少量 OFDM symbol shift
cfg.shift_set = -5:5;

% 如果怀疑 FFT 结果镜像，可设 true；第一轮建议 false
cfg.try_conjugate = false;

% CDM
cfg.cdm_bins = 64;
cfg.cdm_clip = 3.0;

%% ===================== Output dirs =====================

make_dir_local(cfg.out_root);

out_full = fullfile(cfg.out_root, 'first_frame_full');
out_freq = fullfile(cfg.out_root, 'freq_sc');
out_time = fullfile(cfg.out_root, sprintf('time_%d', cfg.M_time));
out_cdm  = fullfile(cfg.out_root, sprintf('cdm_%d', cfg.cdm_bins));
out_log  = fullfile(cfg.out_root, 'logs');
out_snr  = fullfile(cfg.out_root, 'snr_results');

make_dir_local(out_full);
make_dir_local(out_freq);
make_dir_local(out_time);
make_dir_local(out_cdm);
make_dir_local(out_log);
make_dir_local(out_snr);

%% ===================== OFDM struct =====================

ofdm = struct();
ofdm.NumberOfIFFTSamples = cfg.n_fft;
ofdm.NumberOfGuardTime   = cfg.n_guard;
ofdm.Carrier_location    = cfg.carrier_loc;
ofdm.Carrier_location_demo = cfg.carrier_loc_demo;
ofdm.NumberOfCarriers = length(ofdm.Carrier_location);
ofdm.NumberOfCarriers_demo = length(ofdm.Carrier_location_demo);
ofdm.size = cfg.n_syms;

%% ===================== Statistics =====================

index_rows = {};
failure_rows = {};
snr_file_rows = {};

stats = struct();
stats.total_files = 0;
stats.ok_read = 0;
stats.fail_read = 0;
stats.ok_tx_ref = 0;
stats.fail_tx_ref = 0;
stats.ok_deofdm = 0;
stats.fail_deofdm = 0;
stats.ok_snr = 0;
stats.fail_snr = 0;
stats.saved = 0;

all_snr = [];
all_power_sqi = [];
all_mod_labels = {};
all_sub_labels = {};
all_turb_labels = {};

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
    C = TX.QAM.IQmap;

    DSP = struct();
    DSP.DEMAPPER.normMethod = 'MMSE';

    rx_mod_dir = fullfile(cfg.data_root, 'rx_data', cfg.rx_date, mod_name);
    if ~exist(rx_mod_dir, 'dir')
        fprintf('[Skip] Missing RX mod dir: %s\n', rx_mod_dir);
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
            fprintf('[Skip] Missing RX sub dir: %s\n', rx_dir);
            continue;
        end

        bin_list = dir(fullfile(rx_dir, '*.bin'));
        if isempty(bin_list)
            fprintf('[Skip] No .bin in: %s\n', rx_dir);
            continue;
        end

        make_dir_local(fullfile(out_full, mod_name, sub_name));
        make_dir_local(fullfile(out_freq, mod_name, sub_name));
        make_dir_local(fullfile(out_time, mod_name, sub_name));
        make_dir_local(fullfile(out_cdm,  mod_name, sub_name));

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

            %% ---------- A. Load TX references: tx_frame1/2/3 ----------

            tx_refs = cell(1, cfg.n_frames_tx);
            ref_ok = true;
            ref_msg = '';

            for tid = 1:cfg.n_frames_tx

                ref_file = fullfile(cfg.tx_root, mod_name, sub_name, ...
                    sprintf('sig_%04d_frame%d.mat', sig_idx, tid));

                if ~exist(ref_file, 'file')
                    ref_ok = false;
                    ref_msg = ['missing: ' ref_file];
                    break;
                end

                tmp = load(ref_file);

                if ~isfield(tmp, 'data_tx')
                    ref_ok = false;
                    ref_msg = ['no data_tx: ' ref_file];
                    break;
                end

                tx_refs{tid} = tmp.data_tx.';   % [123 × 128]
            end

            if ~ref_ok
                stats.fail_tx_ref = stats.fail_tx_ref + 1;

                failure_rows(end+1,:) = {mod_name, sub_name, turb_name, ...
                    sig_idx, 'tx_ref', ref_msg, rx_bin};

                continue;
            else
                stats.ok_tx_ref = stats.ok_tx_ref + 1;
            end

            %% ---------- B. Read RX .bin as real IM/DD waveform ----------

            try
                [rx80, read_info] = read_keysight_bin_robust_real_local(rx_bin);

                rx80 = rx80(:).';
                rx80 = rx80 - mean(rx80);
                rx80 = rx80 ./ (rms(rx80) + eps);

                stats.ok_read = stats.ok_read + 1;

            catch ME
                stats.fail_read = stats.fail_read + 1;

                failure_rows(end+1,:) = {mod_name, sub_name, turb_name, ...
                    sig_idx, 'read_bin', ME.message, rx_bin};

                continue;
            end

            %% ---------- C. Resample 80G -> 16G, same as rx1 ----------

            try
                rx16 = resample(rx80, cfg.Fs_base, cfg.Fs_rx);

                rx16 = rx16(:).';
                rx16 = rx16 - mean(rx16);

                % 与 rx1 更接近：用 mean(abs(.)) 做幅度归一化
                amp = mean(abs(rx16)) + eps;
                rx16 = rx16 ./ amp;

            catch ME
                failure_rows(end+1,:) = {mod_name, sub_name, turb_name, ...
                    sig_idx, 'resample', ME.message, rx_bin};
                continue;
            end

            %% ---------- D. Use rx1/deOFDM to identify first frame ----------

            try
                rx_flat = deOFDM(rx16, ofdm, cfg.n_syms);

                rx_demo = reshape(rx_flat, cfg.n_syms, ofdm.NumberOfCarriers_demo).';
                rx_sc = rx_demo(1:cfg.n_sc, :);   % [123 × 128]

                stats.ok_deofdm = stats.ok_deofdm + 1;

            catch ME
                stats.fail_deofdm = stats.fail_deofdm + 1;

                failure_rows(end+1,:) = {mod_name, sub_name, turb_name, ...
                    sig_idx, 'deOFDM_first_frame', ME.message, rx_bin};

                continue;
            end

            %% ---------- E. RX1-style SNR: match with tx_frame1/2/3 ----------

            try
                [best_tx_id, snr_frame_rx1_db, snr_sc_rx1_db, ...
                    snr_list_db, best_txafdem, best_ber_sc, align_info] = ...
                    match_rx_to_tx_by_rx1_snr_local(rx_sc, tx_refs, C, DSP, cfg);

                if ~isfinite(snr_frame_rx1_db)
                    error('invalid rx1-style SNR');
                end

                stats.ok_snr = stats.ok_snr + 1;

            catch ME
                stats.fail_snr = stats.fail_snr + 1;

                failure_rows(end+1,:) = {mod_name, sub_name, turb_name, ...
                    sig_idx, 'rx1_style_snr', ME.message, rx_bin};

                continue;
            end

            %% ---------- F. Diagnostics and features ----------

            power_sqi_db = 10 * log10(mean(abs(rx_sc(:)).^2) + eps);

            valid_snr_sc = snr_sc_rx1_db(isfinite(snr_sc_rx1_db));

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

            valid_ber = best_ber_sc(isfinite(best_ber_sc));
            if isempty(valid_ber)
                ber_mean = NaN;
            else
                ber_mean = mean(valid_ber);
            end

            % 时域输入：由于这里只用 rx1 找到第一帧，
            % 这里保存 rx16 的前 M_time 点作为诊断时域样本；
            % 后续真正时域识别如需严格一帧对齐，需要再根据 deOFDM 返回位置改函数。
            if length(rx16) >= cfg.M_time
                rx_time = rx16(1:cfg.M_time);
            else
                rx_time = [rx16, zeros(1, cfg.M_time - length(rx16))];
            end

            cdm64 = make_cdm_from_rxsc_local(rx_sc, cfg.cdm_bins, cfg.cdm_clip);

            %% ---------- G. Save ----------

            base_name = sprintf('sig_%04d_firstframe_rx1', sig_idx);

            full_path = fullfile(out_full, mod_name, sub_name, [base_name '.mat']);
            freq_path = fullfile(out_freq, mod_name, sub_name, [base_name '.mat']);
            time_path = fullfile(out_time, mod_name, sub_name, [base_name '.mat']);
            cdm_path  = fullfile(out_cdm,  mod_name, sub_name, [base_name '.mat']);

            sample = struct();
            sample.rx_sc = single(rx_sc);
            sample.rx_time = single(rx_time);
            sample.cdm64 = single(cdm64);
            sample.txafdem_matrix = single(best_txafdem);
            sample.snr_frame_rx1_db = snr_frame_rx1_db;
            sample.snr_sc_rx1_db = single(snr_sc_rx1_db);
            sample.snr_ref_list_db = snr_list_db;
            sample.best_tx_frame_id = best_tx_id;
            sample.ber_sc = single(best_ber_sc);
            sample.ber_mean = ber_mean;
            sample.power_sqi_db = power_sqi_db;
            sample.align_info = align_info;
            sample.read_info = read_info;

            sample.mod_name = mod_name;
            sample.label_id = label_id;
            sample.mod_order = Mq;
            sample.sub_name = sub_name;
            sample.turbulence = turb_name;
            sample.sig_idx = sig_idx;
            sample.file_key = file_key;
            sample.rx_bin = rx_bin;

            save(full_path, 'sample', '-v7.3');

            sample_freq = struct();
            sample_freq.rx_sc = single(rx_sc);
            sample_freq.label_id = label_id;
            sample_freq.label_name = mod_name;
            sample_freq.file_key = file_key;
            sample_freq.sig_idx = sig_idx;
            sample_freq.best_tx_frame_id = best_tx_id;
            sample_freq.snr_frame_rx1_db = snr_frame_rx1_db;
            save(freq_path, 'sample_freq', '-v7.3');

            sample_time = struct();
            sample_time.rx_time = single(rx_time);
            sample_time.label_id = label_id;
            sample_time.label_name = mod_name;
            sample_time.file_key = file_key;
            sample_time.sig_idx = sig_idx;
            sample_time.snr_frame_rx1_db = snr_frame_rx1_db;
            save(time_path, 'sample_time', '-v7.3');

            sample_cdm = struct();
            sample_cdm.cdm64 = single(cdm64);
            sample_cdm.label_id = label_id;
            sample_cdm.label_name = mod_name;
            sample_cdm.file_key = file_key;
            sample_cdm.sig_idx = sig_idx;
            sample_cdm.snr_frame_rx1_db = snr_frame_rx1_db;
            save(cdm_path, 'sample_cdm', '-v7.3');

            %% ---------- H. Index rows ----------

            index_rows(end+1,:) = { ...
                full_path, freq_path, time_path, cdm_path, ...
                file_key, mod_name, label_id, Mq, sub_name, turb_name, ...
                sig_idx, best_tx_id, snr_frame_rx1_db, ...
                snr_sc_mean_db, snr_sc_std_db, snr_sc_min_db, snr_sc_max_db, ...
                power_sqi_db, ber_mean, ...
                snr_list_db(1), snr_list_db(2), snr_list_db(3), ...
                align_info.shift, align_info.rx_variant};

            snr_file_rows(end+1,:) = { ...
                mod_name, sub_name, turb_name, sig_idx, file_key, ...
                snr_frame_rx1_db, power_sqi_db, ber_mean, best_tx_id};

            all_snr(end+1) = snr_frame_rx1_db;
            all_power_sqi(end+1) = power_sqi_db;
            all_mod_labels{end+1} = mod_name;
            all_sub_labels{end+1} = sub_name;
            all_turb_labels{end+1} = turb_name;

            stats.saved = stats.saved + 1;

            if mod(bi, 15) == 0 || bi == length(bin_list)
                fprintf('%s/%s: %d/%d | saved=%d | fail_read=%d | fail_deofdm=%d | fail_snr=%d\n', ...
                    mod_name, sub_name, bi, length(bin_list), ...
                    stats.saved, stats.fail_read, stats.fail_deofdm, stats.fail_snr);
            end
        end
    end
end

%% ===================== Save tables =====================

index_varnames = { ...
    'FullPath','FreqPath','TimePath','CDMPath', ...
    'FileKey','Mod','LabelID','ModOrder','Sub','Turbulence', ...
    'SigIdx','BestTxFrameID','SNRFrameRX1_dB', ...
    'SNRScMeanRX1_dB','SNRScStdRX1_dB','SNRScMinRX1_dB','SNRScMaxRX1_dB', ...
    'PowerSQI_dB','BERMean', ...
    'SNRwithTx1_dB','SNRwithTx2_dB','SNRwithTx3_dB', ...
    'AlignShift','RXVariant'};

if isempty(index_rows)
    T_index = cell2table(cell(0,length(index_varnames)), ...
        'VariableNames', index_varnames);
else
    T_index = cell2table(index_rows, 'VariableNames', index_varnames);
end

failure_varnames = {'Mod','Sub','Turbulence','SigIdx','Stage','Message','Path'};

if isempty(failure_rows)
    T_fail = cell2table(cell(0,length(failure_varnames)), ...
        'VariableNames', failure_varnames);
else
    T_fail = cell2table(failure_rows, 'VariableNames', failure_varnames);
end

snr_file_varnames = {'Mod','Sub','Turbulence','SigIdx','FileKey', ...
    'SNRFrameRX1_dB','PowerSQI_dB','BERMean','BestTxFrameID'};

if isempty(snr_file_rows)
    T_snr_file = cell2table(cell(0,length(snr_file_varnames)), ...
        'VariableNames', snr_file_varnames);
else
    T_snr_file = cell2table(snr_file_rows, 'VariableNames', snr_file_varnames);
end

T_index = assign_split_by_file_local(T_index, 2026, 0.70, 0.15);

writetable(T_index, fullfile(cfg.out_root, 'dataset_index_firstframe_rx1.csv'));
writetable(T_index, fullfile(cfg.out_root, 'split_index_firstframe_rx1.csv'));
writetable(T_fail,  fullfile(out_log, 'failure_log.csv'));
writetable(T_snr_file, fullfile(out_snr, 'snr_per_file_firstframe_rx1.csv'));

%% ===================== Save summary =====================

snr_summary = struct();
snr_summary.snr_rx1 = all_snr;
snr_summary.power_sqi = all_power_sqi;
snr_summary.mod_labels = all_mod_labels;
snr_summary.sub_labels = all_sub_labels;
snr_summary.turb_labels = all_turb_labels;
snr_summary.stats = stats;
snr_summary.cfg = cfg;

if ~isempty(all_snr)
    snr_summary.mean = mean(all_snr);
    snr_summary.median = median(all_snr);
    snr_summary.std = std(all_snr);
    snr_summary.min = min(all_snr);
    snr_summary.max = max(all_snr);
end

save(fullfile(out_snr, 'snr_summary_firstframe_rx1.mat'), ...
    'snr_summary', '-v7.3');

%% ===================== Print summary =====================

fprintf('\n============================================\n');
fprintf(' First-frame RX1-style SNR Check Complete\n');
fprintf('============================================\n');
fprintf('Total files      : %d\n', stats.total_files);
fprintf('Read OK / Fail   : %d / %d\n', stats.ok_read, stats.fail_read);
fprintf('TX ref OK / Fail : %d / %d\n', stats.ok_tx_ref, stats.fail_tx_ref);
fprintf('deOFDM OK / Fail : %d / %d\n', stats.ok_deofdm, stats.fail_deofdm);
fprintf('SNR OK / Fail    : %d / %d\n', stats.ok_snr, stats.fail_snr);
fprintf('Saved samples    : %d\n', stats.saved);

if ~isempty(all_snr)
    fprintf('\nRX1-style SNR mean/median/std: %.2f / %.2f / %.2f dB\n', ...
        mean(all_snr), median(all_snr), std(all_snr));
    fprintf('RX1-style SNR min/max        : %.2f / %.2f dB\n', ...
        min(all_snr), max(all_snr));
end

fprintf('\nOutput root:\n%s\n', cfg.out_root);

%% ===================== Plot =====================

if ~isempty(all_snr)
    plot_firstframe_rx1_snr_local(all_snr, all_power_sqi, ...
        all_mod_labels, out_snr);
end

fprintf('\n===== DONE =====\n');

%% =====================================================================
%% Helper functions
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

%% ===================== Robust real Keysight reader =====================

function [y, info] = read_keysight_bin_robust_real_local(filename)

    info = struct();

    try
        [y, info] = read_keysight_real_standard_or_infer_local(filename, false);
        info.method = 'standard_bpp_real';
        return;
    catch ME1
        info.standard_error = ME1.message;
    end

    try
        [y, info] = read_keysight_real_standard_or_infer_local(filename, true);
        info.method = 'inferred_bpp_real';
        return;
    catch ME2
        info.infer_error = ME2.message;
    end

    error('read_keysight_bin_robust_real failed: standard=[%s], infer=[%s]', ...
        info.standard_error, info.infer_error);
end

function [y, info] = read_keysight_real_standard_or_infer_local(filename, force_infer)

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

    for ii = 1:length(bpp_candidates)

        bpp = bpp_candidates(ii);

        try
            fseek(fid, data_start, 'bof');

            raw = read_raw_by_bpp_real_local(fid, double(num_points), bpp);
            raw = double(raw(:)).';

            if length(raw) < 1000
                error('raw too short');
            end

            if std(raw(1:min(5000,end))) == 0
                error('zero variance raw');
            end

            % IM/DD APD direct detection: single real waveform.
            y = raw;

            info = struct();
            info.method = 'standard_or_infer_real';
            info.bpp = double(bpp_read);
            info.inferred_bpp = bpp;
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

function raw = read_raw_by_bpp_real_local(fid, num_points, bpp)

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

%% ===================== RX1-style SNR matching =====================

function [best_tx_id, best_snr_db, best_snr_sc_db, ...
    snr_list_db, best_txafdem, best_ber_sc, align_info] = ...
    match_rx_to_tx_by_rx1_snr_local(rx_sc, tx_refs, C, DSP_template, cfg)

    n_ref = length(tx_refs);

    snr_list_db = NaN(1, n_ref);

    best_global_snr = -inf;
    best_tx_id = NaN;
    best_snr_sc_db = NaN(size(rx_sc,1),1);
    best_txafdem = [];
    best_ber_sc = NaN(size(rx_sc,1),1);
    align_info = struct('shift', NaN, 'rx_variant', 'none');

    for tid = 1:n_ref

        tx_ref0 = normalize_tx_ref_shape_for_rx1_local(tx_refs{tid}, size(rx_sc,1));

        if cfg.try_conjugate
            rx_variants = {rx_sc, conj(rx_sc)};
            rx_names = {'normal','conj'};
        else
            rx_variants = {rx_sc};
            rx_names = {'normal'};
        end

        best_tid_snr = -inf;
        best_tid_sc = NaN(size(rx_sc,1),1);
        best_tid_txafdem = [];
        best_tid_ber = NaN(size(rx_sc,1),1);
        best_tid_info = struct('shift', NaN, 'rx_variant', 'none');

        for rv = 1:length(rx_variants)

            rx0 = rx_variants{rv};

            for sh = cfg.shift_set

                [rx_use, tx_use] = align_rx_tx_by_symbol_shift_local(rx0, tx_ref0, sh);

                if isempty(rx_use)
                    continue;
                end

                try
                    [snr_frame_db, snr_sc_db, txafdem_matrix, ber_sc] = ...
                        compute_rx1_style_snr_local(rx_use, tx_use, C, DSP_template);

                    if isfinite(snr_frame_db) && snr_frame_db > best_tid_snr
                        best_tid_snr = snr_frame_db;
                        best_tid_sc = snr_sc_db;
                        best_tid_txafdem = txafdem_matrix;
                        best_tid_ber = ber_sc;
                        best_tid_info.shift = sh;
                        best_tid_info.rx_variant = rx_names{rv};
                    end

                catch
                    continue;
                end
            end
        end

        snr_list_db(tid) = best_tid_snr;

        if isfinite(best_tid_snr) && best_tid_snr > best_global_snr

            best_global_snr = best_tid_snr;
            best_tx_id = tid;
            best_snr_sc_db = best_tid_sc;
            best_txafdem = best_tid_txafdem;
            best_ber_sc = best_tid_ber;
            align_info = best_tid_info;
        end
    end

    best_snr_db = best_global_snr;

    if isempty(best_snr_db) || ~isfinite(best_snr_db)
        best_tx_id = NaN;
        best_snr_db = NaN;
        best_snr_sc_db = NaN(size(rx_sc,1),1);
        best_txafdem = [];
        best_ber_sc = NaN(size(rx_sc,1),1);
        align_info = struct('shift', NaN, 'rx_variant', 'none');
    end
end

function [snr_frame_db, snr_sc_db, txafdem_matrix, ber_sc] = ...
    compute_rx1_style_snr_local(rx_sc, tx_ref, C, DSP_template)

    rx_sc = double(rx_sc);
    tx_ref = double(tx_ref);

    n_sc = min(size(rx_sc,1), size(tx_ref,1));
    n_sym = min(size(rx_sc,2), size(tx_ref,2));

    rx_use = rx_sc(1:n_sc, 1:n_sym);
    tx_use = tx_ref(1:n_sc, 1:n_sym);

    txafdem_matrix = NaN(n_sc, n_sym);
    ber_sc = NaN(n_sc, 1);
    ok_row = false(n_sc, 1);

    for sc = 1:n_sc

        DSP = DSP_template;
        DSP.DEMAPPER.N0 = 0;

        try
            [DSP.DEMAPPER, txafdem] = symDemapper( ...
                rx_use(sc,:), tx_use(sc,:), C, DSP.DEMAPPER);

            txafdem_matrix(sc,:) = txafdem;
            ok_row(sc) = all(isfinite(real(txafdem))) && all(isfinite(imag(txafdem)));

            try
                [BER, ~] = BER_eval(DSP.DEMAPPER.txBits, DSP.DEMAPPER.rxBits);
                ber_sc(sc) = BER;
            catch
                ber_sc(sc) = NaN;
            end

        catch
            ok_row(sc) = false;
        end
    end

    snr_sc_db = NaN(n_sc, 1);
    valid_rows = find(ok_row);

    if isempty(valid_rows)
        snr_frame_db = NaN;
        return;
    end

    try
        [~, snr_tmp] = EVM_eval(rx_use(valid_rows,:), ...
            txafdem_matrix(valid_rows,:));

        snr_tmp = snr_tmp(:);

        n_fill = min(length(valid_rows), length(snr_tmp));
        snr_sc_db(valid_rows(1:n_fill)) = snr_tmp(1:n_fill);

    catch
        for ii = 1:length(valid_rows)

            sc = valid_rows(ii);

            try
                [~, s] = EVM_eval(rx_use(sc,:), txafdem_matrix(sc,:));
                snr_sc_db(sc) = s;
            catch
                snr_sc_db(sc) = NaN;
            end
        end
    end

    valid_snr = snr_sc_db(isfinite(snr_sc_db));

    if isempty(valid_snr)
        snr_frame_db = NaN;
    else
        snr_frame_db = 10 * log10(mean(10.^(valid_snr/10)));
    end
end

function tx_ref_out = normalize_tx_ref_shape_for_rx1_local(tx_ref, n_sc)

    x = tx_ref;

    if size(x,1) == n_sc
        tx_ref_out = x;
    elseif size(x,2) == n_sc
        tx_ref_out = x.';
    else
        x = x(:);
        n_sym = floor(length(x) / n_sc);

        if n_sym < 1
            error('Invalid tx_ref shape');
        end

        x = x(1:n_sc*n_sym);
        tx_ref_out = reshape(x, n_sc, n_sym);
    end
end

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

function T = assign_split_by_file_local(T, seed, train_ratio, val_ratio)

    rng(seed);

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
        n_train = floor(train_ratio * nk);
        n_val = floor(val_ratio * nk);

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

function plot_firstframe_rx1_snr_local(all_snr, all_power_sqi, ...
    all_mod_labels, out_dir)

    figure('Position', [100 100 1400 800]);

    subplot(2,3,1);
    histogram(all_snr, 50, 'FaceColor', '#0072BD', 'EdgeColor', 'none');
    xlabel('RX1-style SNR (dB)');
    ylabel('Count');
    title(sprintf('RX1-style SNR (n=%d, mean=%.2f dB)', ...
        length(all_snr), mean(all_snr)));
    grid on;

    subplot(2,3,2);
    hold on;
    mods = unique(all_mod_labels);
    colors = lines(length(mods));

    for i = 1:length(mods)
        mask = strcmp(all_mod_labels, mods{i});
        histogram(all_snr(mask), 30, ...
            'FaceColor', colors(i,:), ...
            'EdgeColor', 'none', ...
            'FaceAlpha', 0.45);
    end

    legend(mods, 'Location', 'northwest');
    xlabel('RX1-style SNR (dB)');
    ylabel('Count');
    title('RX1-style SNR by Modulation');
    grid on;

    subplot(2,3,3);
    histogram(all_power_sqi, 50, 'FaceColor', '#D95319', 'EdgeColor', 'none');
    xlabel('Power SQI (dB)');
    ylabel('Count');
    title(sprintf('Power SQI (mean=%.2f dB)', mean(all_power_sqi)));
    grid on;

    subplot(2,3,4);
    [f, x] = ecdf(all_snr);
    plot(x, f*100, 'LineWidth', 2);
    xlabel('RX1-style SNR (dB)');
    ylabel('CDF (%)');
    title('Cumulative RX1-style SNR');
    grid on;
    ylim([0 100]);

    subplot(2,3,5);
    boxchart(categorical(all_mod_labels), all_snr);
    xlabel('Modulation');
    ylabel('RX1-style SNR (dB)');
    title('RX1-style SNR Boxplot by Modulation');
    grid on;

    subplot(2,3,6);
    scatter(all_power_sqi, all_snr, 12, 'filled');
    xlabel('Power SQI (dB)');
    ylabel('RX1-style SNR (dB)');
    title('Power SQI vs RX1-style SNR');
    grid on;

    saveas(gcf, fullfile(out_dir, 'firstframe_rx1_snr_summary.png'));
end