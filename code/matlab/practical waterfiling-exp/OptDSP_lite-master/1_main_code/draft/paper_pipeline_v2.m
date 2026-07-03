%% paper_pipeline_v2.m
% 论文级实验信号处理 pipeline — v2 (LTS 双峰帧提取 + 可变帧长)
%
% 依据识别修改.txt 的核心改动：
% 1) 废除固定 frame_len_16 (symbol_bits_16 = 35424)
% 2) LTS 双峰定义帧：Frame_k = LTS_k → LTS_{k+1}
% 3) frame_extract_demod 使用 frame_extract_stable 方案
% 4) 允许可变帧长度：min = payload_only, max = adaptive
%
% 功能：
% 1) 批量读取 Keysight .bin（正确解析文件头）
% 2) 包络相关粗同步 + LTS 双峰精确定义帧边界
% 3) 每帧独立 CFO 补偿、FFT、信道均衡
% 4) SNR 匹配 + 保存完整帧/时域/频域/CDM

clear; clear global; close all; clc;

%% ===================== 0. 路径与全局配置 =====================
addpath(genpath('D:\Project_code\OptDSP_FSO_2\practical waterfiling-exp\OptDSP_lite-master'));

global PROG;
PROG.showMessagesLevel = 1;
initProg();

cfg = struct();
cfg.data_root = 'D:\Project_code\OptDSP_FSO_2\practical waterfiling-exp\OptDSP_lite-master\2_Data_Results';
cfg.rx_date   = '2026.06.26';
cfg.tx_root   = fullfile(cfg.data_root, 'tx_3frame_6mod');
cfg.out_root  = fullfile(cfg.data_root, 'dataset_paper_pipeline_v2');

cfg.Fs_rx   = 80e9;
cfg.Fs_base = 16e9;
cfg.n_frames = 3;
cfg.M_time = 32768;
cfg.cdm_bins = 64;
cfg.cdm_clip = 3.0;

% 处理范围
cfg.mod_names = {'QPSK','16QAM','32QAM','64QAM','128QAM','256QAM'};
cfg.sub_list  = {'sub1','sub2','sub3'};

cfg.turb_map = containers.Map('KeyType','char','ValueType','char');
cfg.turb_map('sub1') = 'weak';
cfg.turb_map('sub2') = 'moderate';
cfg.turb_map('sub3') = 'strong';

% OFDM 参数
cfg.zeros_head = 80;
cfg.n_fft = 256;
cfg.n_guard = 16;
cfg.n_syms = 128;
cfg.carrier_loc = 4:126;
cfg.n_sc = length(cfg.carrier_loc);
cfg.sym_len = cfg.n_fft + cfg.n_guard;
cfg.header_len_16 = cfg.zeros_head + cfg.n_guard + 2*cfg.n_fft;  % 608

% -- v2: 可变帧长参数（识别修改.txt） --
cfg.min_payload_syms = 32;          % 每帧最少 OFDM 符号数

% LTS 双峰检测参数
cfg.lts_search_margin = 3000;       % 帧边界扩展余量（16G 采样点）
cfg.lts_min_peak_dist = 1000;       % LTS 峰最小间距
cfg.lts_peak_thresh_ratio = 0.2;    % 相对峰值阈值

% 同步/对齐参数（80G 粗同步不变）
cfg.sync_decim = 20;
cfg.num_corr_candidates = 60;
cfg.frame_margin_80 = 8000;
cfg.fine_search_len_80 = 30000;
cfg.min_valid_sc_ratio = 0.90;
cfg.min_match_margin_db = 0.30;
cfg.save_rx_frame_80 = false;

% 预加载 LTS
LTS = make_lts_local(cfg.n_fft);

%% ===================== 1. 输出目录与索引文件 =====================
out_full = fullfile(cfg.out_root, 'full_frame_16G');
out_time = fullfile(cfg.out_root, sprintf('time_%d', cfg.M_time));
out_freq = fullfile(cfg.out_root, 'freq_sc');
out_cdm  = fullfile(cfg.out_root, sprintf('cdm_%d', cfg.cdm_bins));
out_log  = fullfile(cfg.out_root, 'logs');
make_dir(out_full); make_dir(out_time); make_dir(out_freq); make_dir(out_cdm); make_dir(out_log);

index_file = fullfile(cfg.out_root, 'index.csv');
failure_file = fullfile(cfg.out_root, 'failure_log.csv');
summary_file = fullfile(cfg.out_root, 'summary.mat');

fid_idx = fopen(index_file, 'w');
fid_fail = fopen(failure_file, 'w');

fprintf(fid_idx, ['out_full,out_time,out_freq,out_cdm,label_id,label_name,mod_order,file_id,sig_idx,' ...
    'rx_frame_idx,best_tx_frame_id,sub_name,turbulence,snr_mean_db,snr_median_db,snr_valid_ratio,' ...
    'snr_match_margin_db,cfo_est,frame_start_80,seg_start_80,seg_end_80,frame_len_80,' ...
    'frame_len_16,lts_peak1_16,lts_peak2_16,sync_metric,valid_flag,valid_reason\n']);

fprintf(fid_fail, 'rx_bin,label_name,sub_name,sig_idx,stage,reason\n');

stats = struct();
stats.total_files = 0;
stats.total_rx_frames_expected = 0;
stats.total_rx_frames_saved = 0;
stats.total_demod_ok = 0;
stats.total_invalid = 0;
stats.fail_no_tx_txt = 0;
stats.fail_no_ref = 0;
stats.fail_sync = 0;
stats.fail_extract = 0;
stats.fail_demod = 0;
stats.fail_snr = 0;

%% ===================== 2. 主循环 =====================
for mi = 1:length(cfg.mod_names)
    mod_name = cfg.mod_names{mi};
    label_id = mi - 1;
    [Mq, bits] = mod_to_order_bits_local(mod_name);
    nBpS_net = bits - 0.2*(bits > 2);

    TX.SIG = setSignalParams('symRate', 8e9, 'M', Mq, ...
        'nPol', 1, 'nBpS', nBpS_net, 'nSyms', cfg.n_syms, ...
        'roll-off', 0.25, 'modulation', 'QAM');
    TX.QAM = QAM_config(TX.SIG);
    C = TX.QAM.IQmap;
    DSP.DEMAPPER.normMethod = 'MMSE';
    DSP.DEMAPPER.normalizeTX = false;

    rx_mod_dir = fullfile(cfg.data_root, 'rx_data', cfg.rx_date, mod_name);
    if ~exist(rx_mod_dir, 'dir')
        fprintf(fid_fail, ',%s,,0,rx_dir_missing,%s\n', mod_name, rx_mod_dir);
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
        if ~exist(rx_dir, 'dir'), continue; end
        bin_list = dir(fullfile(rx_dir, '*.bin'));
        if isempty(bin_list), continue; end

        make_dir(fullfile(out_full, mod_name, turb_name));
        make_dir(fullfile(out_time, mod_name, turb_name));
        make_dir(fullfile(out_freq, mod_name, turb_name));
        make_dir(fullfile(out_cdm,  mod_name, turb_name));

        fprintf('\n========== %s / %s (%s): %d files ==========%s', mod_name, sub_name, turb_name, length(bin_list), newline);

        for bi = 1:length(bin_list)
            [~, fname] = fileparts(bin_list(bi).name);
            sig_idx = str2double(fname);
            if isnan(sig_idx) || sig_idx < 1, continue; end

            rx_bin = fullfile(rx_dir, bin_list(bi).name);
            tx_txt = fullfile(cfg.tx_root, mod_name, sub_name, sprintf('sig_%04d.txt', sig_idx));

            stats.total_files = stats.total_files + 1;
            stats.total_rx_frames_expected = stats.total_rx_frames_expected + cfg.n_frames;

            % ---------- 2.1 读取 RX 80G（正确解析 Keysight 文件头）----------
            try
                rx80 = read_keysight_bin_local(rx_bin);
                rx80 = rx80(:);
                rx80 = rx80 - mean(rx80);
                rx80 = rx80 ./ (rms(rx80) + eps);
            catch ME
                stats.fail_extract = stats.fail_extract + cfg.n_frames;
                log_fail(fid_fail, rx_bin, mod_name, sub_name, sig_idx, 'read_bin', ME.message);
                continue;
            end

            % ---------- 2.2 读取 TX txt ----------
            if ~exist(tx_txt, 'file')
                stats.fail_no_tx_txt = stats.fail_no_tx_txt + cfg.n_frames;
                log_fail(fid_fail, rx_bin, mod_name, sub_name, sig_idx, 'tx_txt_missing', tx_txt);
                continue;
            end
            try
                tx_ref80 = load_ascii_complex_local(tx_txt);
                tx_ref80 = tx_ref80(:);
                frame_len_80 = floor(length(tx_ref80) / cfg.n_frames);
                if frame_len_80 <= 0, error('invalid frame_len_80'); end
            catch ME
                stats.fail_no_tx_txt = stats.fail_no_tx_txt + cfg.n_frames;
                log_fail(fid_fail, rx_bin, mod_name, sub_name, sig_idx, 'tx_txt_read_fail', ME.message);
                continue;
            end

            % ---------- 2.3 加载三个发送参考 data_tx ----------
            tx_refs = cell(1, cfg.n_frames);
            ref_ok = true;
            for tid = 1:cfg.n_frames
                ref_file = fullfile(cfg.tx_root, mod_name, sub_name, sprintf('sig_%04d_frame%d.mat', sig_idx, tid));
                if ~exist(ref_file, 'file')
                    ref_ok = false;
                    log_fail(fid_fail, rx_bin, mod_name, sub_name, sig_idx, 'tx_ref_missing', ref_file);
                    break;
                end
                tmp_ref = load(ref_file);
                if ~isfield(tmp_ref, 'data_tx')
                    ref_ok = false;
                    log_fail(fid_fail, rx_bin, mod_name, sub_name, sig_idx, 'tx_ref_no_data_tx', ref_file);
                    break;
                end
                tx_refs{tid} = tmp_ref.data_tx.';
            end
            if ~ref_ok
                stats.fail_no_ref = stats.fail_no_ref + cfg.n_frames;
                continue;
            end

            % ---------- 2.4 粗同步（80G 包络相关）----------
            try
                [rx_start_80, start_tx_id, sync_metric] = find_awg_cycle_start_local(rx80, tx_ref80, frame_len_80, cfg);
            catch ME
                stats.fail_sync = stats.fail_sync + cfg.n_frames;
                log_fail(fid_fail, rx_bin, mod_name, sub_name, sig_idx, 'sync_fail', ME.message);
                continue;
            end

            % ---------- 2.5 逐帧：LTS双峰提取 → 解调 → SNR匹配 → 保存 ----------
            for rk = 1:cfg.n_frames
                expected_tx_id = mod(start_tx_id + rk - 2, cfg.n_frames) + 1;
                file_id = sprintf('%s_%s_sig%04d', mod_name, sub_name, sig_idx);

                seg_start_80_nom = rx_start_80 + (rk-1) * frame_len_80;

                try
                    % 切 80G 段
                    [rx_raw80, seg_start_80, seg_end_80] = extract_frame_with_margin_local( ...
                        rx80, seg_start_80_nom, frame_len_80, cfg.frame_margin_80);

                    % Resample 到 16G
                    rx_raw16 = resample(rx_raw80(:).', cfg.Fs_base, cfg.Fs_rx);
                    rx_raw16 = rx_raw16 - mean(rx_raw16);
                    rx_raw16 = rx_raw16 ./ (rms(rx_raw16) + eps);

                    % === v2 核心：LTS 双峰帧提取（frame_extract_stable 方案）===
                    frames_cell = frame_extract_stable_local(rx_raw16, LTS.time, cfg);

                    if length(frames_cell) < 1
                        error('frame_extract_stable returned 0 frames');
                    end

                    % 取当前帧（每 80G 段期望取到至少 1 帧）
                    frame_idx_in_seg = min(rk, length(frames_cell));
                    rx_frame16 = frames_cell{frame_idx_in_seg};
                    frame_len_16 = length(rx_frame16);

                    % 解调（适配可变帧长）
                    [rx_sc, cfo_est, demod_info] = demod_one_frame_local(rx_frame16, LTS, cfg);
                catch ME
                    stats.total_invalid = stats.total_invalid + 1;
                    stats.fail_demod = stats.fail_demod + 1;
                    log_fail(fid_fail, rx_bin, mod_name, sub_name, sig_idx, sprintf('frame%d_extract_demod', rk), ME.message);
                    continue;
                end

                % SNR 匹配
                try
                    [best_tx_id, snr_sc, snr_mean_db, snr_median_db, snr_valid_ratio, match_margin_db, match_table] = ...
                        match_tx_reference_and_snr_local(rx_sc, tx_refs, C, DSP, cfg);
                catch ME
                    stats.total_invalid = stats.total_invalid + 1;
                    stats.fail_snr = stats.fail_snr + 1;
                    log_fail(fid_fail, rx_bin, mod_name, sub_name, sig_idx, sprintf('frame%d_snr', rk), ME.message);
                    continue;
                end

                valid_flag = true;
                valid_reason = 'ok';
                if snr_valid_ratio < cfg.min_valid_sc_ratio
                    valid_flag = false;
                    valid_reason = 'low_valid_sc_ratio';
                elseif match_margin_db < cfg.min_match_margin_db
                    valid_reason = 'ambiguous_tx_match';
                end

                % ---------- 保存 ----------
                rx_time = make_time_payload_local(rx_frame16, cfg);
                cdm64 = make_cdm_local(rx_sc, cfg.cdm_bins, cfg.cdm_clip);

                base_name = sprintf('sig_%04d_rxframe%d_tx%d', sig_idx, rk, best_tx_id);
                out_full_file = fullfile(out_full, mod_name, turb_name, [base_name '.mat']);
                out_time_file = fullfile(out_time, mod_name, turb_name, [base_name '.mat']);
                out_freq_file = fullfile(out_freq, mod_name, turb_name, [base_name '.mat']);
                out_cdm_file  = fullfile(out_cdm,  mod_name, turb_name, [base_name '.mat']);

                sample = struct();
                sample.rx_frame_16_full = single(rx_frame16(:));
                if cfg.save_rx_frame_80
                    sample.rx_raw80_with_margin = single(rx_raw80(:));
                end
                sample.label_id = label_id;
                sample.label_name = mod_name;
                sample.mod_order = Mq;
                sample.file_id = file_id;
                sample.sig_idx = sig_idx;
                sample.rx_frame_idx = rk;
                sample.expected_tx_frame_id = expected_tx_id;
                sample.best_tx_frame_id = best_tx_id;
                sample.sub_name = sub_name;
                sample.turbulence = turb_name;
                sample.rx_bin_file = rx_bin;
                sample.Fs_rx = cfg.Fs_rx;
                sample.Fs_base = cfg.Fs_base;
                sample.frame_len_80 = frame_len_80;
                sample.frame_len_16 = frame_len_16;
                sample.frame_start_80 = rx_start_80;
                sample.seg_start_80 = seg_start_80;
                sample.seg_end_80 = seg_end_80;
                sample.sync_metric = sync_metric;
                sample.cfo_est = cfo_est;
                sample.snr_sc = single(snr_sc(:));
                sample.snr_mean_db = snr_mean_db;
                sample.snr_median_db = snr_median_db;
                sample.snr_valid_ratio = snr_valid_ratio;
                sample.snr_match_margin_db = match_margin_db;
                sample.match_table = match_table;
                sample.demod_info = demod_info;
                sample.valid_flag = valid_flag;
                sample.valid_reason = valid_reason;

                sample_time = rmfield(sample, 'rx_frame_16_full');
                sample_time.rx_time = single(rx_time(:));
                sample_freq = rmfield(sample, 'rx_frame_16_full');
                sample_freq.rx_sc = single(rx_sc);
                sample_cdm = rmfield(sample, 'rx_frame_16_full');
                sample_cdm.cdm64 = single(cdm64);

                save(out_full_file, 'sample', '-v7.3');
                save(out_time_file, 'sample_time', '-v7.3');
                save(out_freq_file, 'sample_freq', '-v7.3');
                save(out_cdm_file,  'sample_cdm', '-v7.3');

                fprintf(fid_idx, '%s,%s,%s,%s,%d,%s,%d,%s,%d,%d,%d,%s,%s,%.4f,%.4f,%.4f,%.4f,%.6g,%d,%d,%d,%d,%d,%d,%d,%.6f,%d,%s\n', ...
                    out_full_file, out_time_file, out_freq_file, out_cdm_file, ...
                    label_id, mod_name, Mq, file_id, sig_idx, rk, best_tx_id, sub_name, turb_name, ...
                    snr_mean_db, snr_median_db, snr_valid_ratio, match_margin_db, cfo_est, ...
                    rx_start_80, seg_start_80, seg_end_80, frame_len_80, frame_len_16, ...
                    0, 0, sync_metric, valid_flag, valid_reason);

                stats.total_rx_frames_saved = stats.total_rx_frames_saved + 1;
                if valid_flag
                    stats.total_demod_ok = stats.total_demod_ok + 1;
                else
                    stats.total_invalid = stats.total_invalid + 1;
                end
            end
        end
    end
end

%% ===================== 3. 收尾 =====================
fclose(fid_idx);
fclose(fid_fail);

fprintf('\n========== Pipeline v2 Complete ==========%s', newline);
fprintf('Total files:            %d%s', stats.total_files, newline);
fprintf('Frames expected:        %d%s', stats.total_rx_frames_expected, newline);
fprintf('Frames saved:           %d%s', stats.total_rx_frames_saved, newline);
fprintf('  valid (SNR ok):       %d%s', stats.total_demod_ok, newline);
fprintf('  invalid:              %d%s', stats.total_invalid, newline);
fprintf('Fail (no tx_txt):       %d%s', stats.fail_no_tx_txt, newline);
fprintf('Fail (no ref):          %d%s', stats.fail_no_ref, newline);
fprintf('Fail (sync):            %d%s', stats.fail_sync, newline);
fprintf('Fail (extract/demod):   %d%s', stats.fail_demod, newline);
fprintf('Fail (SNR):             %d%s', stats.fail_snr, newline);

save(summary_file, 'stats');
fprintf('Summary saved to: %s%s', summary_file, newline);

%% =====================================================================
%%                       辅助函数
%% =====================================================================

function make_dir(d)
    if ~exist(d, 'dir'), mkdir(d); end
end

function log_fail(fid, rx_bin, mod_name, sub_name, sig_idx, stage, reason)
    fprintf(fid, '%s,%s,%s,%d,%s,%s\n', rx_bin, mod_name, sub_name, sig_idx, stage, reason);
end

function [Mq, bits] = mod_to_order_bits_local(mod_name)
    switch mod_name
        case 'QPSK',    Mq = 4;   bits = 2;
        case '16QAM',   Mq = 16;  bits = 4;
        case '32QAM',   Mq = 32;  bits = 5;
        case '64QAM',   Mq = 64;  bits = 6;
        case '128QAM',  Mq = 128; bits = 7;
        case '256QAM',  Mq = 256; bits = 8;
        otherwise, error('unknown mod: %s', mod_name);
    end
end

%% ===== 正确读 Keysight .bin（batch_preprocess.m 版本）=====
function y = read_keysight_bin_local(filename)
    fid = fopen(filename, 'rb');
    if fid == -1, error('Cannot open: %s', filename); end
    fread(fid, 2, '*char')'; fread(fid, 2, '*char')';
    fread(fid, 1, 'int32'); fread(fid, 1, 'int32');
    fread(fid, 1, 'int32'); fread(fid, 1, 'int32');
    fread(fid, 1, 'int32'); num_points = fread(fid, 1, 'int32');
    fread(fid, 1, 'int32'); fread(fid, 1, 'float32');
    fread(fid, 1, 'float64'); fread(fid, 1, 'float64');
    fread(fid, 1, 'float64'); fread(fid, 1, 'int32');
    fread(fid, 1, 'int32'); fread(fid, 16, '*char')';
    fread(fid, 16, '*char')'; fread(fid, 24, '*char')';
    fread(fid, 16, '*char')'; fread(fid, 1, 'float64');
    fread(fid, 1, 'uint32'); fread(fid, 1, 'int32');
    fread(fid, 1, 'int16'); bpp = fread(fid, 1, 'int16');
    fread(fid, 1, 'int32');
    switch bpp
        case 4, raw = fread(fid, num_points, 'float32');
        case 2, raw = fread(fid, num_points, 'int16');
        case 1, raw = fread(fid, num_points, 'int8');
        otherwise, raw = fread(fid, num_points, 'double');
    end
    fclose(fid);
    % interleaved I/Q → complex
    raw = raw(:);
    n = floor(length(raw)/2);
    raw = raw(1:2*n);
    y = double(raw(1:2:end)) + 1j * double(raw(2:2:end));
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
    LTS_f = LongTrainSym_ini(1:n_fft);
    LTS_f([1 n_fft/2+1]) = 0;
    ltrs_in = LTS_f;
    ltrs_in(1, n_fft/2+2:n_fft) = conj(ltrs_in(1, n_fft/2:-1:2));
    LTS.freq = LTS_f(:);
    LTS.time = ifft(ltrs_in(:));
end

%% ===== 粗同步：80G 包络相关（同 v1）=====
function [start_80, start_tx_id, metric_best] = find_awg_cycle_start_local(rx80, tx_ref80, frame_len_80, cfg)
    rx80 = rx80(:); tx_ref80 = tx_ref80(:);
    if length(tx_ref80) < cfg.n_frames * frame_len_80
        error('tx_ref80 too short');
    end
    if length(rx80) < cfg.n_frames * frame_len_80
        error('rx80 too short');
    end
    decim = cfg.sync_decim;
    rx_env = abs(rx80(1:decim:end));
    rx_env = normalize_vec_local(rx_env);
    all_score = []; all_idx = []; all_tid = [];
    for tid = 1:cfg.n_frames
        tx_frame = tx_ref80((tid-1)*frame_len_80 + (1:frame_len_80));
        tx_env = abs(tx_frame(1:decim:end));
        max_len = min(length(tx_env), 12000);
        tx_env = normalize_vec_local(tx_env(1:max_len));
        if length(rx_env) < length(tx_env), continue; end
        c = conv(rx_env, flipud(tx_env), 'valid');
        c_abs = abs(c);
        n_take = min(cfg.num_corr_candidates, length(c_abs));
        [vals, idxs] = maxk(c_abs, n_take);
        all_score = [all_score; vals(:)]; all_idx = [all_idx; idxs(:)];
        all_tid = [all_tid; tid * ones(n_take,1)];
    end
    if isempty(all_score), error('no correlation candidate'); end
    [~, order] = sort(all_score, 'descend');
    for ii = 1:length(order)
        idx_d = all_idx(order(ii)); tid = all_tid(order(ii));
        coarse_start = (idx_d - 1)*decim + 1;
        start_candidate = refine_start_fullrate_env_local(rx80, tx_ref80, frame_len_80, tid, coarse_start, cfg);
        tx_id_candidate = tid;
        while start_candidate + cfg.n_frames*frame_len_80 - 1 > length(rx80)
            start_candidate = start_candidate - frame_len_80;
            tx_id_candidate = mod(tx_id_candidate - 2, cfg.n_frames) + 1;
        end
        while start_candidate < 1
            start_candidate = start_candidate + frame_len_80;
            tx_id_candidate = mod(tx_id_candidate, cfg.n_frames) + 1;
        end
        if start_candidate >= 1 && start_candidate + cfg.n_frames*frame_len_80 - 1 <= length(rx80)
            start_80 = round(start_candidate); start_tx_id = tx_id_candidate;
            metric_best = all_score(order(ii)); return;
        end
    end
    error('cannot contain 3 complete frames');
end

function start_refined = refine_start_fullrate_env_local(rx80, tx_ref80, frame_len_80, tx_id, coarse_start, cfg)
    tpl = tx_ref80((tx_id-1)*frame_len_80 + (1:frame_len_80));
    L = min([length(tpl), cfg.fine_search_len_80, length(rx80)]);
    tpl_env = normalize_vec_local(abs(tpl(1:L)));
    win = 4 * cfg.sync_decim;
    s1 = max(1, coarse_start - win);
    s2 = min(length(rx80) - L + 1, coarse_start + win);
    if s2 < s1, start_refined = coarse_start; return; end
    best_val = -inf; best_s = coarse_start;
    for s = s1:s2
        r_env = normalize_vec_local(abs(rx80(s:s+L-1)));
        val = abs(r_env(:)' * tpl_env(:));
        if val > best_val, best_val = val; best_s = s; end
    end
    start_refined = best_s;
end

function v = normalize_vec_local(v)
    v = v(:);
    v = v - mean(v);
    v = v ./ (std(v) + eps);
end

function [rx_raw80, seg_start, seg_end] = extract_frame_with_margin_local(rx80, seg_start_nom, frame_len_80, margin)
    seg_start = max(1, seg_start_nom - margin);
    seg_end = min(length(rx80), seg_start_nom + frame_len_80 - 1 + margin);
    if seg_end <= seg_start, error('invalid segment range'); end
    rx_raw80 = rx80(seg_start:seg_end);
end

%% ===== v2 核心：LTS 双峰帧提取（识别修改.txt → frame_extract_stable）=====
function frames = frame_extract_stable_local(rx, LTS_time, cfg)
    % 按识别修改.txt 的 frame_extract_stable 方案
    % Frame_k = LTS_k → LTS_{k+1}
    rx = rx(:).';
    n_fft = cfg.n_fft;
    search_margin = cfg.lts_search_margin;
    min_peak_dist = cfg.lts_min_peak_dist;
    peak_thresh_ratio = cfg.lts_peak_thresh_ratio;

    L = length(LTS_time);

    % Step 1: LTS 滑动相关
    xc = abs(conv(rx, flipud(conj(LTS_time)), 'valid'));

    % Step 2: 找所有 LTS 峰
    peak_thresh = peak_thresh_ratio * max(xc);
    [~, locs] = findpeaks(xc, 'MinPeakHeight', peak_thresh, ...
                          'MinPeakDistance', min_peak_dist);

    if length(locs) < 2
        error('Not enough LTS peaks detected: %d', length(locs));
    end

    % Step 3: 估计周期（AWG 循环关键）
    deltas = diff(locs);
    frame_period = median(deltas);

    % Step 4: 构建帧（Frame_k = LTS_k → LTS_{k+1}）
    frames = {};
    for k = 1:length(locs) - 1
        s = locs(k);
        e = locs(k+1);

        % 防异常短帧
        if (e - s) < 0.5 * frame_period
            continue;
        end

        % 自适应扩展边界
        s_ext = max(1, s - search_margin);
        e_ext = min(length(rx), e + search_margin);

        frame = rx(s_ext:e_ext);
        frame = frame - mean(frame);
        frame = frame ./ (rms(frame) + eps);

        frames{end+1} = frame;
    end

    if isempty(frames)
        error('frame_extract_stable: no valid frames constructed');
    end
end

%% ===== v2 适配：解调（帧从 LTS 前 search_margin 开始）=====
function [rx_sc, cfo, info] = demod_one_frame_local(rx_frame16, LTS, cfg)
    rx = rx_frame16(:).';
    n_fft = cfg.n_fft;
    n_guard = cfg.n_guard;
    sym_len = cfg.sym_len;

    % 在扩展帧中找 LTS1 的位置（LTS峰在 search_margin 附近）
    LTS_t = LTS.time(:).';
    xc = abs(conv(rx, flipud(conj(LTS_t)), 'valid'));
    [~, lts_peak] = max(xc);

    % 从帧内找到真正的帧起始
    frm_start = lts_peak;  % LTS1 位置 = 帧起始
    remaining = length(rx) - frm_start + 1;

    if remaining < cfg.header_len_16
        error('frame too short after LTS alignment: %d < %d', remaining, cfg.header_len_16);
    end

    % 取 LTS1 和 LTS2
    lts1 = rx(frm_start : frm_start + n_fft - 1);
    lts2 = rx(frm_start + n_fft : frm_start + 2*n_fft - 1);

    if length(lts1) < n_fft || length(lts2) < n_fft
        error('frame too short for LTS');
    end

    % CFO 补偿
    cfo = angle(sum(lts1(:).*conj(lts2(:))))/(2*pi*n_fft);
    % 从帧起始开始 CFO 补偿
    rx_comp = rx(frm_start:end) .* exp(-1j*2*pi*cfo*(0:remaining-1)/n_fft);

    % 数据从 LTS2 之后开始
    data_start = 2 * n_fft + 1;
    dp = rx_comp(data_start:end);
    nd = floor(length(dp) / sym_len);

    if nd < cfg.min_payload_syms
        error('not enough OFDM symbols: %d < %d', nd, cfg.min_payload_syms);
    end

    dp = dp(1:nd*sym_len);
    dm = reshape(dp, sym_len, nd);
    dn = dm(n_guard+1:end, :);
    fd = fft(dn, n_fft, 1) / sqrt(n_fft);

    % 信道估计
    lts_avg = (lts1(:) + lts2(:)) / 2;
    lts_fd = fft(lts_avg, n_fft) / sqrt(n_fft);
    H = lts_fd ./ (LTS.freq(:) + 1e-12);
    H(abs(LTS.freq(:)) < 0.5) = 1;
    feq = fd ./ H;

    rx_sc = feq(cfg.carrier_loc, :);
    info = struct();
    info.nd = nd;
    info.n_syms_actual = nd;
    info.lts_peak_in_frame = lts_peak;
    info.H = single(H);
end

%% ===== v2 适配：参考匹配（兼容可变符号数）=====
function [best_tx_id, best_snr_sc, best_mean_db, best_median_db, valid_ratio, margin_db, match_table] = ...
    match_tx_reference_and_snr_local(rx_sc, tx_refs, C, DSP_in, cfg)

    n_ref = length(tx_refs);
    mean_db = NaN(1, n_ref);
    median_db = NaN(1, n_ref);
    valid_ratio_list = NaN(1, n_ref);
    snr_cell = cell(1, n_ref);

    for tid = 1:n_ref
        tx_ref = tx_refs{tid};
        n_sym = min(size(rx_sc, 2), size(tx_ref, 2));
        if n_sym < cfg.min_payload_syms, continue; end
        rx_use = rx_sc(:, 1:n_sym);
        tx_use = tx_ref(:, 1:n_sym);
        DSP = DSP_in;

        C_vec = C(:);
        SNR_sc = NaN(cfg.n_sc, 1);
        for sc = 1:cfg.n_sc
            rx_flat = rx_use(sc, :).';
            if all(isnan(rx_flat)) || all(isinf(rx_flat)), continue; end
            dist2 = abs(rx_flat - C_vec.').^2;
            [~, idx] = min(dist2, [], 2);
            decisions = C_vec(idx);
            h = (rx_flat' * decisions) / (rx_flat' * rx_flat + eps);
            rx_scaled = rx_flat * h;
            err = rx_scaled - decisions;
            sig_pwr = mean(abs(decisions).^2);
            noi_pwr = mean(abs(err).^2);
            if noi_pwr > 0 && sig_pwr > 0 && isfinite(noi_pwr)
                SNR_sc(sc) = 10 * log10(sig_pwr / noi_pwr);
            end
        end
        SNR_sc = SNR_sc(:);
        valid = SNR_sc(isfinite(SNR_sc) & SNR_sc > 0);
        snr_cell{tid} = SNR_sc;
        valid_ratio_list(tid) = length(valid) / cfg.n_sc;
        if ~isempty(valid)
            mean_db(tid) = 10*log10(mean(10.^(valid/10)));
            median_db(tid) = median(valid);
        end
    end

    [best_mean_db, best_tx_id] = max(mean_db);
    if ~isfinite(best_mean_db), error('all tx reference matches have invalid SNR'); end
    best_snr_sc = snr_cell{best_tx_id};
    best_median_db = median_db(best_tx_id);
    valid_ratio = valid_ratio_list(best_tx_id);

    sorted = sort(mean_db(isfinite(mean_db)), 'descend');
    margin_db = NaN; if length(sorted) >= 2, margin_db = sorted(1) - sorted(2); end

    match_table = struct();
    match_table.mean_snr_db = mean_db;
    match_table.median_snr_db = median_db;
    match_table.valid_ratio = valid_ratio_list;
end

function rx_time = make_time_payload_local(rx_frame16, cfg)
    payload_start = cfg.header_len_16 + 1;
    rx = rx_frame16(:);
    if payload_start > length(rx)
        rx_time = zeros(cfg.M_time, 1);
        return;
    end
    if payload_start + cfg.M_time - 1 <= length(rx)
        rx_time = rx(payload_start:payload_start+cfg.M_time-1);
    else
        rx_time = rx(payload_start:end);
        rx_time = [rx_time; zeros(cfg.M_time - length(rx_time), 1)];
    end
    rx_time = rx_time - mean(rx_time);
    rx_time = rx_time ./ (rms(rx_time) + eps);
end

function CDM = make_cdm_local(rx_sc, nbin, clip_val)
    z = rx_sc(:);
    z = z(isfinite(real(z)) & isfinite(imag(z)));
    if isempty(z), CDM = zeros(nbin, nbin); return; end
    z = z - mean(z);
    z = z ./ (rms(abs(z)) + eps);
    zr = real(z); zi = imag(z);
    zr = max(min(zr, clip_val), -clip_val);
    zi = max(min(zi, clip_val), -clip_val);
    edges = linspace(-clip_val, clip_val, nbin+1);
    H = histcounts2(zi, zr, edges, edges);
    CDM = log1p(H);
    CDM = CDM ./ (max(CDM(:)) + eps);
end
