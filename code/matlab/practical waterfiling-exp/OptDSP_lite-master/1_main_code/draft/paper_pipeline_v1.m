%% paper_pipeline_v1.m
% 论文级实验信号处理 pipeline
%
% 功能：
% 1) 批量读取 Keysight .bin 原始 80G 接收长序列；
% 2) 根据 tx_3frame_6mod 中保存的 sig_XXXX.txt 得到真实 AWG 单帧周期 frame_len_80；
% 3) 用三帧发送波形的包络相关进行离线数据集对齐，自动处理 AWG 123123... 循环起点；
% 4) 在 80G 原始采样率下切出连续 3 帧，再对每帧单独 resample 到 16G；
% 5) 在单帧内使用 LTS 做细同步、CFO 粗补偿、FFT、信道均衡，得到 123×128 频域符号；
% 6) 每个 RX 帧分别与 tx_frame1/2/3 的 data_tx 匹配，选择最佳 tx_frame_id，计算子载波 SNR；
% 7) 保存：完整帧、32768 时域净载荷、频域子载波、64×64 CDM 星座密度图、index.csv 和 failure_log.csv。
%
% 重要说明：
% - 该脚本用于“离线数据集构建、质量筛查和 SNR 标注”。
% - 发送端参考 tx_txt / data_tx 只用于离线对齐和 SNR 评估，不允许作为最终识别模型输入。
% - 最终 Python 训练建议只读取 rx_time / rx_sc / cdm64 和 label，不读取 data_tx、BER、EVM 等参考辅助量。

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
cfg.out_root  = fullfile(cfg.data_root, 'dataset_paper_pipeline_v1');

cfg.Fs_rx   = 80e9;
cfg.Fs_base = 16e9;
cfg.n_frames = 3;
cfg.M_time = 32768;
cfg.cdm_bins = 64;
cfg.cdm_clip = 3.0;

% 处理范围
cfg.mod_names = {'QPSK','16QAM','32QAM','64QAM','128QAM','256QAM'};
cfg.sub_list  = {'sub1','sub2','sub3'};   % 如需 sub4，可加入 'sub4'

% sub 到实验条件映射；按你的实验含义修改
cfg.turb_map = containers.Map('KeyType','char','ValueType','char');
cfg.turb_map('sub1') = 'weak';
cfg.turb_map('sub2') = 'moderate';
cfg.turb_map('sub3') = 'strong';

% OFDM 参数，与 tx3 保持一致
cfg.zeros_head = 80;
cfg.n_fft = 256;
cfg.n_guard = 16;
cfg.n_syms = 128;
cfg.carrier_loc = 4:126;
cfg.n_sc = length(cfg.carrier_loc);
cfg.sym_len = cfg.n_fft + cfg.n_guard;
cfg.symbol_bits_16 = cfg.zeros_head + cfg.n_guard + 2*cfg.n_fft + cfg.sym_len*cfg.n_syms;  % 35424
cfg.header_len_16 = cfg.zeros_head + cfg.n_guard + 2*cfg.n_fft;                              % 608

% 同步/对齐参数
cfg.sync_decim = 20;              % 80G 包络相关降采样倍数
cfg.num_corr_candidates = 60;     % 每个文件最多尝试的相关候选起点数量
cfg.frame_margin_80 = 8000;       % 每帧切片时前后冗余，重采样后再靠 LTS 精确裁剪
cfg.fine_search_len_80 = 30000;   % 相关细化时使用的全分辨率模板长度
cfg.min_valid_sc_ratio = 0.90;    % 有效SNR子载波比例阈值
cfg.min_match_margin_db = 0.30;   % 最佳tx参考与第二佳参考SNR差，低于该值标记为ambiguous但仍保存

% 是否保存体积较大的 80G 原始帧；一般不建议保存
cfg.save_rx_frame_80 = false;

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

fprintf(fid_idx, ['out_full,out_time,out_freq,out_cdm,label_id,label_name,mod_order,file_id,sig_idx,', ...
    'rx_frame_idx,best_tx_frame_id,sub_name,turbulence,snr_mean_db,snr_median_db,snr_valid_ratio,', ...
    'snr_match_margin_db,cfo_est,frame_start_80,seg_start_80,seg_end_80,frame_len_80,', ...
    'sync_metric,lts_peak_16,valid_flag,valid_reason\n']);

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

% 预先加载 LTS
LTS = make_lts_local(cfg.n_fft);

%% ===================== 2. 主循环：调制格式 / sub / bin =====================
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
        if ~exist(rx_dir, 'dir')
            continue;
        end
        bin_list = dir(fullfile(rx_dir, '*.bin'));
        if isempty(bin_list)
            continue;
        end

        make_dir(fullfile(out_full, mod_name, turb_name));
        make_dir(fullfile(out_time, mod_name, turb_name));
        make_dir(fullfile(out_freq, mod_name, turb_name));
        make_dir(fullfile(out_cdm,  mod_name, turb_name));

        fprintf('\n========== %s / %s (%s): %d files ==========%s', mod_name, sub_name, turb_name, length(bin_list), newline);

        for bi = 1:length(bin_list)
            [~, fname] = fileparts(bin_list(bi).name);
            sig_idx = str2double(fname);
            if isnan(sig_idx) || sig_idx < 1
                continue;
            end

            rx_bin = fullfile(rx_dir, bin_list(bi).name);
            tx_txt = fullfile(cfg.tx_root, mod_name, sub_name, sprintf('sig_%04d.txt', sig_idx));

            stats.total_files = stats.total_files + 1;
            stats.total_rx_frames_expected = stats.total_rx_frames_expected + cfg.n_frames;

            % ---------- 2.1 读取 RX 80G ----------
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

            % ---------- 2.2 读取 TX txt，确定真实 80G 帧长 ----------
            if ~exist(tx_txt, 'file')
                stats.fail_no_tx_txt = stats.fail_no_tx_txt + cfg.n_frames;
                log_fail(fid_fail, rx_bin, mod_name, sub_name, sig_idx, 'tx_txt_missing', tx_txt);
                continue;
            end

            try
                tx_ref80 = load_ascii_complex_local(tx_txt);
                tx_ref80 = tx_ref80(:);
                frame_len_80 = floor(length(tx_ref80) / cfg.n_frames);
                if frame_len_80 <= 0
                    error('invalid frame_len_80');
                end
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
                tx_refs{tid} = tmp_ref.data_tx.';  % [123 x 128]
            end
            if ~ref_ok
                stats.fail_no_ref = stats.fail_no_ref + cfg.n_frames;
                continue;
            end

            % ---------- 2.4 相关同步：找到一个能完整取 3 帧的 AWG 周期起点 ----------
            try
                [rx_start_80, start_tx_id, sync_metric] = find_awg_cycle_start_local(rx80, tx_ref80, frame_len_80, cfg);
            catch ME
                stats.fail_sync = stats.fail_sync + cfg.n_frames;
                log_fail(fid_fail, rx_bin, mod_name, sub_name, sig_idx, 'sync_fail', ME.message);
                continue;
            end

            % ---------- 2.5 逐帧切分、解调、参考匹配、保存 ----------
            for rk = 1:cfg.n_frames
                expected_tx_id = mod(start_tx_id + rk - 2, cfg.n_frames) + 1;
                file_id = sprintf('%s_%s_sig%04d', mod_name, sub_name, sig_idx);

                seg_start_80_nom = rx_start_80 + (rk-1) * frame_len_80;
                seg_end_80_nom   = seg_start_80_nom + frame_len_80 - 1;

                try
                    [rx_raw80, seg_start_80, seg_end_80] = extract_frame_with_margin_local( ...
                        rx80, seg_start_80_nom, frame_len_80, cfg.frame_margin_80);

                    rx_raw16 = resample(rx_raw80(:).', cfg.Fs_base, cfg.Fs_rx);
                    rx_raw16 = rx_raw16 - mean(rx_raw16);
                    rx_raw16 = rx_raw16 ./ (rms(rx_raw16) + eps);

                    [rx_frame16, lts_peak_16, local_start_16] = align_single_frame_by_lts_local(rx_raw16, LTS, cfg);

                    [rx_sc, cfo_est, demod_info] = demod_one_frame_local(rx_frame16, LTS, cfg);
                catch ME
                    stats.total_invalid = stats.total_invalid + 1;
                    stats.fail_demod = stats.fail_demod + 1;
                    log_fail(fid_fail, rx_bin, mod_name, sub_name, sig_idx, sprintf('frame%d_extract_demod', rk), ME.message);
                    continue;
                end

                % 参考匹配：当前 RX 帧分别和 tx_frame1/2/3 计算 SNR，选最佳
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
                    % 不直接丢弃，但标记为参考匹配不够明确。后续训练可根据 valid_reason 决定是否筛除。
                    valid_reason = 'ambiguous_tx_match';
                end

                % ---------- 保存样本 ----------
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
                sample.frame_start_80 = rx_start_80;
                sample.seg_start_80 = seg_start_80;
                sample.seg_end_80 = seg_end_80;
                sample.sync_metric = sync_metric;
                sample.lts_peak_16 = lts_peak_16;
                sample.local_start_16 = local_start_16;
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

                fprintf(fid_idx, '%s,%s,%s,%s,%d,%s,%d,%s,%d,%d,%d,%s,%s,%.4f,%.4f,%.4f,%.4f,%.6g,%d,%d,%d,%d,%.6f,%d,%d,%s\n', ...
                    out_full_file, out_time_file, out_freq_file, out_cdm_file, ...
                    label_id, mod_name, Mq, file_id, sig_idx, rk, best_tx_id, sub_name, turb_name, ...
                    snr_mean_db, snr_median_db, snr_valid_ratio, match_margin_db, cfo_est, ...
                    rx_start_80, seg_start_80, seg_end_80, frame_len_80, sync_metric, lts_peak_16, valid_flag, valid_reason);

                stats.total_rx_frames_saved = stats.total_rx_frames_saved + 1;
                if valid_flag
                    stats.total_demod_ok = stats.total_demod_ok + 1;
                else
                    stats.total_invalid = stats.total_invalid + 1;
                end
            end

            if mod(bi, 10) == 0 || bi == length(bin_list)
                fprintf('[%s/%s] %d/%d files, saved=%d, valid=%d\n', ...
                    mod_name, sub_name, bi, length(bin_list), stats.total_rx_frames_saved, stats.total_demod_ok);
            end
        end
    end
end

fclose(fid_idx);
fclose(fid_fail);

save(summary_file, 'cfg', 'stats', '-v7.3');

fprintf('\n========================================\n');
fprintf('论文级 pipeline 完成\n');
fprintf('处理文件数: %d\n', stats.total_files);
fprintf('理论帧数:   %d\n', stats.total_rx_frames_expected);
fprintf('保存帧数:   %d\n', stats.total_rx_frames_saved);
fprintf('有效帧数:   %d\n', stats.total_demod_ok);
fprintf('无效/警告:  %d\n', stats.total_invalid);
fprintf('index.csv:  %s\n', index_file);
fprintf('failure:    %s\n', failure_file);
fprintf('summary:    %s\n', summary_file);
fprintf('========================================\n');

%% ========================================================================
%% 局部函数
%% ========================================================================

function make_dir(d)
    if ~exist(d, 'dir')
        mkdir(d);
    end
end

function log_fail(fid, rx_bin, mod_name, sub_name, sig_idx, stage, reason)
    reason = strrep(reason, ',', ';');
    fprintf(fid, '%s,%s,%s,%d,%s,%s\n', rx_bin, mod_name, sub_name, sig_idx, stage, reason);
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

function y = read_keysight_bin_local(filename)
    fid = fopen(filename, 'rb');
    if fid == -1
        error('Cannot open: %s', filename);
    end
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
        case 4
            y = fread(fid, num_points, 'float32').';
        case 2
            y = fread(fid, num_points, 'int16').';
        case 1
            y = fread(fid, num_points, 'int8').';
        otherwise
            y = fread(fid, num_points, 'double').';
    end
    fclose(fid);
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

function [start_80, start_tx_id, metric_best] = find_awg_cycle_start_local(rx80, tx_ref80, frame_len_80, cfg)
    rx80 = rx80(:);
    tx_ref80 = tx_ref80(:);
    if length(tx_ref80) < cfg.n_frames * frame_len_80
        error('tx_ref80 length is shorter than 3*frame_len_80');
    end
    if length(rx80) < cfg.n_frames * frame_len_80
        error('rx80 is shorter than 3 frames');
    end

    decim = cfg.sync_decim;
    rx_env = abs(rx80(1:decim:end));
    rx_env = normalize_vec_local(rx_env);

    all_score = [];
    all_idx = [];
    all_tid = [];

    for tid = 1:cfg.n_frames
        tx_frame = tx_ref80((tid-1)*frame_len_80 + (1:frame_len_80));
        tx_env = abs(tx_frame(1:decim:end));
        max_len = min(length(tx_env), 12000);
        tx_env = normalize_vec_local(tx_env(1:max_len));

        if length(rx_env) < length(tx_env)
            continue;
        end
        c = conv(rx_env, flipud(tx_env), 'valid');
        c_abs = abs(c);
        n_take = min(cfg.num_corr_candidates, length(c_abs));
        [vals, idxs] = maxk(c_abs, n_take);

        all_score = [all_score; vals(:)]; %#ok<AGROW>
        all_idx   = [all_idx; idxs(:)]; %#ok<AGROW>
        all_tid   = [all_tid; tid * ones(n_take,1)]; %#ok<AGROW>
    end

    if isempty(all_score)
        error('no correlation candidate found');
    end
    [~, order] = sort(all_score, 'descend');

    metric_best = NaN;
    for ii = 1:length(order)
        idx_d = all_idx(order(ii));
        tid = all_tid(order(ii));
        coarse_start = (idx_d - 1)*decim + 1;
        start_candidate = refine_start_fullrate_env_local(rx80, tx_ref80, frame_len_80, tid, coarse_start, cfg);
        tx_id_candidate = tid;

        % 如果从该点向后放不下3帧，就向前回退整数帧；同时更新对应tx帧id
        while start_candidate + cfg.n_frames*frame_len_80 - 1 > length(rx80)
            start_candidate = start_candidate - frame_len_80;
            tx_id_candidate = mod(tx_id_candidate - 2, cfg.n_frames) + 1;  % 1->3, 2->1, 3->2
        end
        while start_candidate < 1
            start_candidate = start_candidate + frame_len_80;
            tx_id_candidate = mod(tx_id_candidate, cfg.n_frames) + 1;      % 1->2, 2->3, 3->1
        end

        if start_candidate >= 1 && start_candidate + cfg.n_frames*frame_len_80 - 1 <= length(rx80)
            start_80 = round(start_candidate);
            start_tx_id = tx_id_candidate;
            metric_best = all_score(order(ii));
            return;
        end
    end
    error('all correlation candidates cannot contain 3 complete frames');
end

function start_refined = refine_start_fullrate_env_local(rx80, tx_ref80, frame_len_80, tx_id, coarse_start, cfg)
    tpl = tx_ref80((tx_id-1)*frame_len_80 + (1:frame_len_80));
    L = min([length(tpl), cfg.fine_search_len_80, length(rx80)]);
    tpl_env = normalize_vec_local(abs(tpl(1:L)));

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
        r_env = normalize_vec_local(abs(rx80(s:s+L-1)));
        val = abs(r_env(:)' * tpl_env(:));
        if val > best_val
            best_val = val;
            best_s = s;
        end
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
    if seg_end <= seg_start
        error('invalid segment range');
    end
    rx_raw80 = rx80(seg_start:seg_end);
end

function [rx_frame16, lts_peak, frm_start] = align_single_frame_by_lts_local(rx_raw16, LTS, cfg)
    rx = rx_raw16(:).';
    n_fft = cfg.n_fft;
    if length(rx) < cfg.symbol_bits_16
        error('resampled raw frame shorter than symbol_bits_16');
    end
    if length(rx) < n_fft
        error('raw16 too short for LTS search');
    end

    LTS_t = LTS.time(:).';
    xc_len = length(rx) - n_fft + 1;
    xc = zeros(1, xc_len);
    for ni = 1:xc_len
        xc(ni) = abs(sum(rx(ni:ni+n_fft-1) .* conj(LTS_t)));
    end
    [~, lts_peak] = max(xc);

    % 如果帧从1开始，LTS起点应为 zeros_head+n_guard+1，所以 frm_start = lts_peak - zeros_head - n_guard
    frm_start = lts_peak - cfg.zeros_head - cfg.n_guard;
    if frm_start < 1
        error('LTS peak too close to segment beginning: frm_start=%d', frm_start);
    end
    frm_end = frm_start + cfg.symbol_bits_16 - 1;
    if frm_end > length(rx)
        error('aligned 16G frame incomplete: start=%d end=%d len=%d', frm_start, frm_end, length(rx));
    end
    rx_frame16 = rx(frm_start:frm_end);
    rx_frame16 = rx_frame16 - mean(rx_frame16);
    rx_frame16 = rx_frame16 ./ (rms(rx_frame16) + eps);
end

function [rx_sc, cfo, info] = demod_one_frame_local(rx_frame16, LTS, cfg)
    rx = rx_frame16(:).';
    n_fft = cfg.n_fft;
    n_guard = cfg.n_guard;
    sym_len = cfg.sym_len;

    if length(rx) < cfg.symbol_bits_16
        error('rx_frame16 shorter than full frame');
    end

    lts1 = rx(cfg.zeros_head+n_guard+1 : cfg.zeros_head+n_guard+n_fft);
    lts2 = rx(cfg.zeros_head+n_guard+n_fft+1 : cfg.zeros_head+n_guard+2*n_fft);

    cfo = angle(sum(lts1(:).*conj(lts2(:))))/(2*pi*n_fft);
    rx = rx .* exp(-1j*2*pi*cfo*(0:length(rx)-1)/n_fft);

    ds = cfg.zeros_head + n_guard + 2*n_fft + 1;
    dp = rx(ds:end);
    nd = floor(length(dp)/sym_len);
    if nd < cfg.n_syms
        error('not enough OFDM symbols: nd=%d', nd);
    end
    dp = dp(1:cfg.n_syms*sym_len);
    dm = reshape(dp, sym_len, cfg.n_syms);
    dn = dm(n_guard+1:end, :);
    fd = fft(dn, n_fft, 1)/sqrt(n_fft);

    lts_avg = (lts1(:) + lts2(:))/2;
    lts_fd = fft(lts_avg, n_fft)/sqrt(n_fft);
    H = lts_fd ./ (LTS.freq(:) + 1e-12);
    H(abs(LTS.freq(:)) < 0.5) = 1;
    feq = fd ./ H;

    rx_sc = feq(cfg.carrier_loc, :);  % [123 x 128]
    info = struct();
    info.nd = nd;
    info.H = single(H);
end

function [best_tx_id, best_snr_sc, best_mean_db, best_median_db, valid_ratio, margin_db, match_table] = ...
    match_tx_reference_and_snr_local(rx_sc, tx_refs, C, DSP_in, cfg)

    n_ref = length(tx_refs);
    mean_db = NaN(1, n_ref);
    median_db = NaN(1, n_ref);
    valid_ratio_list = NaN(1, n_ref);
    snr_cell = cell(1, n_ref);

    for tid = 1:n_ref
        tx_ref = tx_refs{tid};
        n_sym = min(size(rx_sc,2), size(tx_ref,2));
        rx_use = rx_sc(:, 1:n_sym);
        tx_use = tx_ref(:, 1:n_sym);
        DSP = DSP_in;

        C_vec = C(:);
        SNR_sc = NaN(cfg.n_sc, 1);
        for sc = 1:cfg.n_sc
            rx_flat = rx_use(sc, :).';
            if all(isnan(rx_flat)) || all(isinf(rx_flat))
                continue;
            end
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
    if ~isfinite(best_mean_db)
        error('all tx reference matches have invalid SNR');
    end
    best_snr_sc = snr_cell{best_tx_id};
    best_median_db = median_db(best_tx_id);
    valid_ratio = valid_ratio_list(best_tx_id);

    sorted = sort(mean_db(isfinite(mean_db)), 'descend');
    if length(sorted) >= 2
        margin_db = sorted(1) - sorted(2);
    else
        margin_db = NaN;
    end

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
    if isempty(z)
        CDM = zeros(nbin, nbin);
        return;
    end
    z = z - mean(z);
    z = z ./ (rms(abs(z)) + eps);
    zr = real(z);
    zi = imag(z);
    zr = max(min(zr, clip_val), -clip_val);
    zi = max(min(zi, clip_val), -clip_val);
    edges = linspace(-clip_val, clip_val, nbin+1);
    H = histcounts2(zi, zr, edges, edges);  % 行: Q, 列: I
    CDM = log1p(H);
    CDM = CDM ./ (max(CDM(:)) + eps);
end
