%% paper_pipeline_v2_test.m
% 基于 修改1.txt → frame_extract_stable_v2
% LTS 双峰 + 周期估计 + 自适应帧边界

clear; clear global; close all; clc;

addpath(genpath('D:\Project_code\OptDSP_FSO_2\practical waterfiling-exp\OptDSP_lite-master'));

global PROG;
PROG.showMessagesLevel = 0;
initProg();

%% ===================== 配置 =====================
cfg = struct();
cfg.data_root = 'D:\Project_code\OptDSP_FSO_2\practical waterfiling-exp\OptDSP_lite-master\2_Data_Results';
cfg.rx_date   = '2026.06.26';
cfg.tx_root   = fullfile(cfg.data_root, 'tx_3frame_6mod');

cfg.Fs_rx   = 80e9;
cfg.Fs_base = 16e9;
cfg.n_frames = 3;
cfg.M_time = 32768;

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
cfg.header_len_16 = cfg.zeros_head + cfg.n_guard + 2*cfg.n_fft;

% 帧提取参数（修改1.txt）
cfg.lts_search_margin = 3000;

% 同步参数
cfg.sync_decim = 20;
cfg.num_corr_candidates = 60;
cfg.frame_margin_80 = 8000;
cfg.fine_search_len_80 = 30000;

LTS = make_lts_local(cfg.n_fft);

%% ===================== 统计 =====================
result_cell = {};
err_counts = containers.Map();

stats = struct();
stats.total_bin_files = 0;
stats.total_rx_frames  = 0;
stats.ok_read          = 0;
stats.ok_tx_txt        = 0;
stats.ok_ref           = 0;
stats.ok_sync          = 0;
stats.ok_extract_demod = 0;
stats.fail_read_bin    = 0;
stats.fail_tx_txt      = 0;
stats.fail_ref         = 0;
stats.fail_sync        = 0;
stats.fail_files       = 0;

%% ===================== 主循环 =====================
for mi = 1:length(cfg.mod_names)
    mod_name = cfg.mod_names{mi};
    [Mq, ~] = mod_to_order_bits_local(mod_name);
    nBpS_net = (log2(Mq)) - 0.2*((log2(Mq)) > 2);

    TX.SIG = setSignalParams('symRate', 8e9, 'M', Mq, ...
        'nPol', 1, 'nBpS', nBpS_net, 'nSyms', cfg.n_syms, ...
        'roll-off', 0.25, 'modulation', 'QAM');
    TX.QAM = QAM_config(TX.SIG);
    C = TX.QAM.IQmap;
    DSP.DEMAPPER.normMethod = 'MMSE';
    DSP.DEMAPPER.normalizeTX = false;

    rx_mod_dir = fullfile(cfg.data_root, 'rx_data', cfg.rx_date, mod_name);
    if ~exist(rx_mod_dir, 'dir'), continue; end

    for si = 1:length(cfg.sub_list)
        sub_name = cfg.sub_list{si};

        rx_dir = fullfile(rx_mod_dir, sub_name);
        if ~exist(rx_dir, 'dir'), continue; end
        bin_list = dir(fullfile(rx_dir, '*.bin'));
        if isempty(bin_list), continue; end

        for bi = 1:length(bin_list)
            [~, fname] = fileparts(bin_list(bi).name);
            sig_idx = str2double(fname);
            if isnan(sig_idx) || sig_idx < 1, continue; end

            rx_bin = fullfile(rx_dir, bin_list(bi).name);
            tx_txt = fullfile(cfg.tx_root, mod_name, sub_name, sprintf('sig_%04d.txt', sig_idx));

            R = struct();
            R.mod = mod_name; R.sub = sub_name; R.sig_idx = sig_idx;
            R.n_frames_ok = 0; R.fail_stage = ''; R.fail_reason = '';
            R.frame_lens = NaN(cfg.n_frames, 1);

            stats.total_bin_files = stats.total_bin_files + 1;
            stats.total_rx_frames  = stats.total_rx_frames + cfg.n_frames;

            % ---- read_bin ----
            try
                rx80 = read_keysight_bin_local(rx_bin);
                rx80 = rx80(:);
                rx80 = rx80 - mean(rx80);
                rx80 = rx80 ./ (rms(rx80) + eps);
                stats.ok_read = stats.ok_read + 1;
            catch ME
                stats.fail_read_bin = stats.fail_read_bin + 1;
                stats.fail_files = stats.fail_files + 1;
                R.fail_stage = 'read_bin';
                R.fail_reason = ME.message;
                result_cell{end+1} = R;
                continue;
            end

            % ---- tx_txt ----
            if ~exist(tx_txt, 'file')
                stats.fail_tx_txt = stats.fail_tx_txt + 1;
                stats.fail_files = stats.fail_files + 1;
                R.fail_stage = 'tx_txt_missing';
                result_cell{end+1} = R;
                continue;
            end
            try
                tx_ref80 = load_ascii_complex_local(tx_txt);
                tx_ref80 = tx_ref80(:);
                frame_len_80 = floor(length(tx_ref80) / cfg.n_frames);
                if frame_len_80 <= 0, error('invalid'); end
                stats.ok_tx_txt = stats.ok_tx_txt + 1;
            catch ME
                stats.fail_tx_txt = stats.fail_tx_txt + 1;
                stats.fail_files = stats.fail_files + 1;
                R.fail_stage = 'tx_txt_read_fail';
                R.fail_reason = ME.message;
                result_cell{end+1} = R;
                continue;
            end

            % ---- tx_refs ----
            tx_refs = cell(1, cfg.n_frames);
            ref_ok = true;
            for tid = 1:cfg.n_frames
                ref_file = fullfile(cfg.tx_root, mod_name, sub_name, ...
                    sprintf('sig_%04d_frame%d.mat', sig_idx, tid));
                if ~exist(ref_file, 'file'), ref_ok = false; break; end
                tmp_ref = load(ref_file);
                if ~isfield(tmp_ref, 'data_tx'), ref_ok = false; break; end
                tx_refs{tid} = tmp_ref.data_tx.';
            end
            if ~ref_ok
                stats.fail_ref = stats.fail_ref + 1;
                stats.fail_files = stats.fail_files + 1;
                R.fail_stage = 'tx_ref_missing';
                result_cell{end+1} = R;
                continue;
            end
            stats.ok_ref = stats.ok_ref + 1;

            % ---- sync ----
            try
                [rx_start_80, ~, ~] = find_awg_cycle_start_local( ...
                    rx80, tx_ref80, frame_len_80, cfg);
                stats.ok_sync = stats.ok_sync + 1;
            catch ME
                stats.fail_sync = stats.fail_sync + 1;
                stats.fail_files = stats.fail_files + 1;
                R.fail_stage = 'sync_fail';
                R.fail_reason = ME.message;
                result_cell{end+1} = R;
                continue;
            end

            % ---- LTS 双峰帧提取（修改1.txt 的 frame_extract_stable_v2）----
            frames_ok = 0;
            frame_lens = NaN(cfg.n_frames, 1);

            for rk = 1:cfg.n_frames
                seg_start_nom = rx_start_80 + (rk-1) * frame_len_80;
                try
                    [rx_raw80, ~, ~] = extract_frame_with_margin_local( ...
                        rx80, seg_start_nom, frame_len_80, cfg.frame_margin_80);

                    rx_raw16 = resample(rx_raw80(:).', cfg.Fs_base, cfg.Fs_rx);
                    rx_raw16 = rx_raw16 - mean(rx_raw16);
                    rx_raw16 = rx_raw16 ./ (rms(rx_raw16) + eps);

                    % frame_extract_stable_v2
                    frames_cell = frame_extract_stable_v2(rx_raw16, LTS.time, cfg.lts_search_margin);

                    if isempty(frames_cell)
                        error('frame_extract_stable_v2 returned 0 frames');
                    end

                    frame_idx = min(rk, length(frames_cell));
                    rx_frame16 = frames_cell{frame_idx};
                    rx_frame16 = rx_frame16 - mean(rx_frame16);
                    rx_frame16 = rx_frame16 ./ (rms(rx_frame16) + eps);

                    % 解调
                    [rx_sc, ~, ~] = demod_one_frame_local_simple(rx_frame16, LTS, cfg);

                    frames_ok = frames_ok + 1;
                    frame_lens(rk) = length(rx_frame16);
                catch ME
                    if isKey(err_counts, ME.message)
                        err_counts(ME.message) = err_counts(ME.message) + 1;
                    else
                        err_counts(ME.message) = 1;
                    end
                end
            end

            stats.ok_extract_demod = stats.ok_extract_demod + frames_ok;
            R.n_frames_ok = frames_ok;
            R.frame_lens = frame_lens;
            if frames_ok < cfg.n_frames
                stats.fail_files = stats.fail_files + 1;
                R.fail_stage = sprintf('extract_demod: %d/%d ok', frames_ok, cfg.n_frames);
            end
            result_cell{end+1} = R;

            if mod(bi, 20) == 0 || bi == length(bin_list)
                fprintf('%s/%s: %d/%d\n', mod_name, sub_name, bi, length(bin_list));
            end
        end
    end
end

%% ===================== 结果 =====================
fprintf('\n============================================\n');
fprintf('  frame_extract_stable_v2 测试结果\n');
fprintf('============================================\n');
fprintf('总 .bin 文件数:      %d\n', stats.total_bin_files);
fprintf('总期望帧数:          %d\n', stats.total_rx_frames);
fprintf('--------------------------------------------\n');
fprintf('read_bin   OK: %d  | FAIL: %d\n', stats.ok_read,       stats.fail_read_bin);
fprintf('tx_txt     OK: %d  | FAIL: %d\n', stats.ok_tx_txt,     stats.fail_tx_txt);
fprintf('tx_ref     OK: %d  | FAIL: %d\n', stats.ok_ref,        stats.fail_ref);
fprintf('sync       OK: %d  | FAIL: %d\n', stats.ok_sync,       stats.fail_sync);
fprintf('extract/demod OK: %d  | FAIL: %d\n', ...
    stats.ok_extract_demod, stats.total_rx_frames - stats.ok_extract_demod);
fprintf('--------------------------------------------\n');
rate = stats.ok_extract_demod / stats.total_rx_frames * 100;
fprintf('帧提取成功率: %.2f%% (%d/%d)\n', rate, stats.ok_extract_demod, stats.total_rx_frames);

fprintf('\n--- extract_demod 错误统计 ---\n');
if err_counts.Count > 0
    keys_list = keys(err_counts);
    vals_list = values(err_counts);
    [~, order] = sort(cell2mat(vals_list), 'descend');
    for i = 1:min(15, length(order))
        ki = order(i);
        fprintf('  [%d次] %s\n', vals_list{ki}, strrep(keys_list{ki}, newline, ' '));
    end
    if length(order) > 15
        fprintf('  ... 以及其他 %d 种错误\n', length(order) - 15);
    end
else
    fprintf('  (无错误)\n');
end
fprintf('\n===== 测试结束 =====\n');

%% =====================================================================
%%                       辅助函数
%% =====================================================================

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

%% ===== 正确读 Keysight .bin =====
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

%% ===== 粗同步：80G 包络相关 =====
function [start_80, start_tx_id, metric_best] = find_awg_cycle_start_local(rx80, tx_ref80, frame_len_80, cfg)
    rx80 = rx80(:); tx_ref80 = tx_ref80(:);
    if length(tx_ref80) < cfg.n_frames * frame_len_80, error('tx_ref80 too short'); end
    if length(rx80) < cfg.n_frames * frame_len_80, error('rx80 too short'); end
    decim = cfg.sync_decim;
    rx_env = abs(rx80(1:decim:end)); rx_env = normalize_vec_local(rx_env);
    all_score = []; all_idx = []; all_tid = [];
    for tid = 1:cfg.n_frames
        tx_frame = tx_ref80((tid-1)*frame_len_80 + (1:frame_len_80));
        tx_env = abs(tx_frame(1:decim:end));
        max_len = min(length(tx_env), 12000);
        tx_env = normalize_vec_local(tx_env(1:max_len));
        if length(rx_env) < length(tx_env), continue; end
        c = conv(rx_env, flipud(tx_env), 'valid'); c_abs = abs(c);
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
    v = v(:); v = v - mean(v); v = v ./ (std(v) + eps);
end

function [rx_raw80, seg_start, seg_end] = extract_frame_with_margin_local(rx80, seg_start_nom, frame_len_80, margin)
    seg_start = max(1, seg_start_nom - margin);
    seg_end = min(length(rx80), seg_start_nom + frame_len_80 - 1 + margin);
    if seg_end <= seg_start, error('invalid segment range'); end
    rx_raw80 = rx80(seg_start:seg_end);
end

%% ===== frame_extract_stable_v2（修改1.txt 原文）=====
function frames = frame_extract_stable_v2(rx, LTS_time, search_margin)
    % Stable Frame Extraction for AWG cyclic OFDM signal
    % Based on: LTS correlation, periodicity estimation, adaptive frame boundary
    rx = rx(:).';
    if nargin < 3, search_margin = 3000; end

    % 1. LTS matched filter
    L = length(LTS_time);
    xc = abs(conv(rx, flipud(conj(LTS_time)), 'valid'));
    [~, locs] = findpeaks(xc, ...
        'MinPeakHeight', 0.5*max(xc), ...
        'MinPeakDistance', L);

    if length(locs) < 2
        error('LTS detection failed: not enough peaks (%d)', length(locs));
    end

    % 2. estimate frame period
    deltas = diff(locs);
    frame_period = round(median(deltas));

    % 3. remove abnormal peaks
    valid_locs = locs;
    valid_locs = valid_locs([true diff(valid_locs) > 0.5*frame_period]);

    % 4. build frames adaptively
    frames = {};
    k = 1;
    while k < length(valid_locs)
        start_p = valid_locs(k);
        end_p   = valid_locs(k+1);

        if (end_p - start_p) < 0.5 * frame_period
            k = k + 1;
            continue;
        end

        s = max(1, start_p - search_margin);
        e = min(length(rx), end_p + search_margin);

        frames{end+1} = rx(s:e);
        k = k + 1;
    end

    if isempty(frames)
        error('frame_extract_stable_v2: no valid frames (peaks=%d, period=%d)', ...
              length(valid_locs), frame_period);
    end
end

%% ===== 解调（从 LTS 峰开始）=====
function [rx_sc, cfo, info] = demod_one_frame_local_simple(rx_frame16, LTS, cfg)
    rx = rx_frame16(:).';
    n_fft = cfg.n_fft; n_guard = cfg.n_guard; sym_len = cfg.sym_len;

    LTS_t = LTS.time(:).';
    xc = abs(conv(rx, flipud(conj(LTS_t)), 'valid'));
    [~, lts_peak] = max(xc);

    frm_start = lts_peak;
    remaining = length(rx) - frm_start + 1;

    if remaining < 2 * n_fft
        error('frame too short for LTS: remaining=%d', remaining);
    end

    lts1 = rx(frm_start : frm_start + n_fft - 1);
    lts2 = rx(frm_start + n_fft : frm_start + 2*n_fft - 1);

    cfo = angle(sum(lts1(:).*conj(lts2(:))))/(2*pi*n_fft);
    rx_comp = rx(frm_start:end) .* exp(-1j*2*pi*cfo*(0:remaining-1)/n_fft);

    data_start = 2 * n_fft + 1;
    dp = rx_comp(data_start:end);
    nd = floor(length(dp) / sym_len);

    % 允许 0 个符号（测试不报错）
    if nd < 1
        error('no complete OFDM symbols: nd=%d', nd);
    end

    dp = dp(1:nd*sym_len);
    dm = reshape(dp, sym_len, nd);
    dn = dm(n_guard+1:end, :);
    fd = fft(dn, n_fft, 1) / sqrt(n_fft);

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
