%% rx_uniform_qam_test_sub03.m — 均匀 QAM 接收测试（强湍流 sub03）
% =========================================================
% 基于 test_one_rx_repeat_rx1_sync_3frames.m 修改
% 适配均匀 QAM（无 CCDM）数据格式
%
% 功能：
%   1. 读取 TX 参考：uniformQAM_6mod/{mod}/sub{01-10}/sig_XXXX.mat
%   2. 读取 RX .bin 文件（需先完成 FSO 传输）
%   3. IM/DD 实值读取 → 80G→16G → rx1-style 同步
%   4. deOFDM + DSP 均衡
%   5. 每帧与 TX 参考匹配，计算 SNR
%   6. 画出星座图，检查调制格式是否正确
%   7. 验证 16QAM→4×4, 32QAM→十字, 64QAM→8×8, 128QAM→十字, 256QAM→16×16
%
% 用法：
%   1. 先运行 tx_uniform_qam_6mod.m 生成 TX 信号
%   2. 用 AWG 发送 TX .txt 文件，经 FSO 信道（sub03 强湍流）后接收
%   3. 运行本脚本进行接收处理
% =========================================================

clear; clear global; close all; clc;

addpath(genpath('D:\Project_code\OptDSP_FSO_2\practical waterfiling-exp\OptDSP_lite-master'));

global PROG;
PROG.showMessagesLevel = 1;
initProg();

%% ===================== 用户配置区 =====================

cfg = struct();

cfg.data_root = 'D:\Project_code\OptDSP_FSO_2\practical waterfiling-exp\OptDSP_lite-master\2_Data_Results';

% TX 参考数据目录（均匀 QAM）
cfg.tx_root = fullfile(cfg.data_root, 'uniformQAM_6mod');

% RX 数据目录（FSO 传输后产生的 .bin 文件）
% TODO: 修改为实际日期
cfg.rx_date = '2026.06.28';
cfg.rx_sub  = 'sub03';        % 强湍流

% 测试配置：每种调制测试第一个 bin（3帧）
cfg.test_bin = 1;             % bin 编号 1~10

cfg.Fs_rx   = 80e9;
cfg.Fs_base = 16e9;

cfg.n_frames = 3;

% 调制格式列表
cfg.mod_names = {'QPSK','16QAM','32QAM','64QAM','128QAM','256QAM'};

% OFDM 参数
cfg.zeros_head = 80;
cfg.n_fft      = 256;
cfg.n_guard    = 16;
cfg.n_syms     = 128;

cfg.carrier_loc = 4:126;
cfg.carrier_loc_demo = [4:126, 132:254];

cfg.n_sc = length(cfg.carrier_loc);
cfg.n_sc_demo = length(cfg.carrier_loc_demo);

cfg.sym_len = cfg.n_fft + cfg.n_guard;

% 理论 16G 帧长
cfg.frame_len_16 = cfg.zeros_head + cfg.n_guard + 2*cfg.n_fft + cfg.sym_len*cfg.n_syms;

% deOFDM 中相当于 fine_index - (zeros_head + LtrsCPLength - 5)
cfg.frame_pre_lts = cfg.zeros_head + cfg.n_guard - 5;

% 下一帧搜索时跳过当前帧的回退余量
cfg.next_search_backoff = 800;

% rx1-style SNR 匹配时允许少量 OFDM 符号偏移
cfg.shift_set = -5:5;

% 第一轮建议 false；如果 SNR 异常低，再改 true 看是否有镜像/共轭问题
cfg.try_conjugate = false;

cfg.plot_en = true;

%% ===================== OFDM struct =====================

ofdm = struct();
ofdm.NumberOfIFFTSamples = cfg.n_fft;
ofdm.NumberOfGuardTime = cfg.n_guard;
ofdm.Carrier_location = cfg.carrier_loc;
ofdm.Carrier_location_demo = cfg.carrier_loc_demo;
ofdm.NumberOfCarriers = cfg.n_sc;
ofdm.NumberOfCarriers_demo = cfg.n_sc_demo;
ofdm.size = cfg.n_syms;

%% ===================== LTS =====================

LTS = make_lts_local(cfg.n_fft);

%% ===================== 存储所有调制结果 =====================

all_results = {};

%% ===================== 主循环：遍历每种调制格式 =====================

for mi = 1:length(cfg.mod_names)

    mod_name = cfg.mod_names{mi};

    fprintf('\n');
    fprintf('============================================\n');
    fprintf(' Processing: %s\n', mod_name);
    fprintf('============================================\n');

    %% ---------- QAM 参数（均匀 QAM，无 CCDM 开销） ----------

    [Mq, bits] = mod_to_order_bits_local(mod_name);

    % 均匀 QAM：nBpS_net = bits（不扣除 CCDM 开销）
    nBpS_net = bits;

    TX.SIG = setSignalParams('symRate', 8e9, 'M', Mq, ...
        'nPol', 1, 'nBpS', nBpS_net, 'nSyms', cfg.n_syms, ...
        'roll-off', 0.25, 'modulation', 'QAM');

    TX.QAM = QAM_config(TX.SIG);
    C = TX.QAM.IQmap;

    DSP = struct();
    DSP.DEMAPPER.normMethod = 'MMSE';

    %% ---------- 加载 TX 参考（3帧） ----------

    tx_refs = cell(1, cfg.n_frames);
    ref_ok = true;

    bin_str = sprintf('sub%02d', cfg.test_bin);

    for tid = 1:cfg.n_frames

        sig_id = (cfg.test_bin - 1) * cfg.n_frames + tid;
        ref_file = fullfile(cfg.tx_root, mod_name, bin_str, ...
            sprintf('sig_%04d.mat', sig_id));

        if ~exist(ref_file, 'file')
            warning('TX reference missing: %s', ref_file);
            ref_ok = false;
            break;
        end

        tmp = load(ref_file);

        if ~isfield(tmp, 'data_tx')
            warning('No data_tx in: %s', ref_file);
            ref_ok = false;
            break;
        end

        tx_refs{tid} = tmp.data_tx.';   % [123 × 128]
        fprintf('Loaded TX ref frame%d: %s\n', tid, ref_file);
    end

    if ~ref_ok
        warning('Skipping %s: TX reference load failed.', mod_name);
        continue;
    end

    %% ---------- 读取 RX .bin ----------

    rx_bin = fullfile(cfg.data_root, 'rx_data', cfg.rx_date, ...
        mod_name, cfg.rx_sub, sprintf('%d.bin', cfg.test_bin));

    fprintf('\n[1] Reading RX .bin...\n');
    fprintf('  File: %s\n', rx_bin);

    if ~exist(rx_bin, 'file')
        warning('RX .bin missing: %s', rx_bin);
        fprintf('  → 请先完成 FSO 传输（AWG 发送 → 湍流信道 → 示波器采集）\n');
        continue;
    end

    try
        [rx80, read_info] = read_keysight_bin_robust_real_local(rx_bin);

        rx80 = rx80(:).';
        rx80 = rx80 - mean(rx80);
        rx80 = rx80 ./ (rms(rx80) + eps);

        fprintf('  Read OK. RX80 length = %d\n', length(rx80));
        fprintf('  Reader method: %s, inferred bpp=%g\n', ...
            read_info.method, read_info.inferred_bpp);

    catch ME
        warning('Failed to read RX .bin: %s', ME.message);
        continue;
    end

    %% ---------- 80G → 16G ----------

    fprintf('\n[2] Resample %.0f GSa/s -> %.0f GSa/s...\n', ...
        cfg.Fs_rx/1e9, cfg.Fs_base/1e9);

    rx16 = resample(rx80, cfg.Fs_base, cfg.Fs_rx);
    rx16 = rx16(:).';

    rx16 = rx16 - mean(rx16);
    rx16 = rx16 ./ (mean(abs(rx16)) + eps);

    fprintf('  RX16 length = %d\n', length(rx16));

    % AWG 循环保护：把开头拼到末尾
    wrap_len = min(length(rx16), 3*cfg.frame_len_16);
    rx16_ext = [rx16, rx16(1:wrap_len)];

    fprintf('  RX16 extended length = %d\n', length(rx16_ext));

    %% ---------- 重复 rx1-style 同步，提取 3 帧 ----------

    cursor = 1;
    results = struct();
    results.frames = [];

    fprintf('\n[3] Repeat rx1-style sync for 3 frames...\n');

    for rk = 1:cfg.n_frames

        fprintf('\n  --- Frame %d/%d ---\n', rk, cfg.n_frames);
        fprintf('  Search cursor = %d\n', cursor);

        if cursor >= length(rx16_ext) - cfg.frame_len_16
            warning('  Cursor too close to end. Stop.');
            break;
        end

        search_sig = rx16_ext(cursor:end);

        %% ------ A. rx1-style 同步 ------

        try
            [lts_start_rel, frame_start_rel, sync_info] = ...
                find_one_frame_start_rx1_style_local(search_sig, ofdm, cfg);

            lts_start_abs   = cursor + lts_start_rel - 1;
            frame_start_abs = cursor + frame_start_rel - 1;

            fprintf('  Sync OK: edge=%d, fine=%d, lts_abs=%d, frame_abs=%d\n', ...
                sync_info.edge_index, sync_info.fine_time_est, ...
                lts_start_abs, frame_start_abs);

        catch ME
            warning('  Frame %d sync failed: %s', rk, ME.message);
            break;
        end

        %% ------ B. 从 LTS 起点解调这一帧 ------

        try
            [rx_sc, rx_time, rx_frame16_lts, demod_info] = ...
                demod_one_frame_from_lts_start_local(rx16_ext, lts_start_abs, LTS, cfg);

            fprintf('  Demod OK: n_use=%d, CFO=%.4e\n', ...
                demod_info.n_use, demod_info.cfo);

        catch ME
            warning('  Frame %d demod failed: %s', rk, ME.message);
            cursor = cursor + round(0.8 * cfg.frame_len_16);
            continue;
        end

        %% ------ C. 与 TX 参考匹配，计算 SNR ------

        try
            [best_tx_id, best_snr_db, best_snr_sc_db, ...
                snr_list_db, best_txafdem, best_ber_sc, align_info] = ...
                match_rx_to_tx_by_rx1_snr_local(rx_sc, tx_refs, C, DSP, cfg);

            fprintf('  SNR with tx1/tx2/tx3 = %.2f / %.2f / %.2f dB\n', ...
                snr_list_db(1), snr_list_db(2), snr_list_db(3));
            fprintf('  Best TX frame = %d, Best SNR = %.2f dB\n', ...
                best_tx_id, best_snr_db);

        catch ME
            warning('  Frame %d SNR matching failed: %s', rk, ME.message);
            best_tx_id = NaN;
            best_snr_db = NaN;
            best_snr_sc_db = NaN(cfg.n_sc,1);
            snr_list_db = NaN(1,3);
            best_txafdem = [];
            best_ber_sc = NaN(cfg.n_sc,1);
            align_info = struct('shift', NaN, 'rx_variant', 'none');
        end

        %% ------ D. 保存结果 ------

        one = struct();
        one.rx_frame_idx = rk;
        one.rx_sc = rx_sc;
        one.rx_time = rx_time;
        one.best_tx_id = best_tx_id;
        one.snr_frame_rx1_db = best_snr_db;
        one.snr_sc_rx1_db = best_snr_sc_db;
        one.snr_list_db = snr_list_db;
        one.txafdem_matrix = best_txafdem;
        one.ber_sc = best_ber_sc;

        results.frames = [results.frames, one];

        %% ------ E. 星座图 ------

        if cfg.plot_en
            figure('Name', sprintf('%s Frame %d', mod_name, rk), ...
                'Position', [100+80*rk, 100+50*rk, 1000, 420]);

            subplot(1,2,1);
            plot(rx_sc(:), 'b.');
            axis square; grid on;
            title(sprintf('%s Frame %d: rx\\_sc (equalized)', mod_name, rk));
            xlabel('I'); ylabel('Q');

            subplot(1,2,2);
            if ~isempty(best_txafdem)
                plot(best_txafdem(:), 'r.');
                axis square; grid on;
                title(sprintf('%s Frame %d: Decision, SNR=%.2f dB', ...
                    mod_name, rk, best_snr_db));
                xlabel('I'); ylabel('Q');
            else
                text(0.1, 0.5, 'No decision');
                axis off;
            end
        end

        %% ------ F. 跳过当前帧，找下一帧 ------

        cursor_next = frame_start_abs + cfg.frame_len_16 - cfg.next_search_backoff;

        if cursor_next <= cursor
            cursor_next = cursor + round(0.8 * cfg.frame_len_16);
        end

        cursor = cursor_next;
    end

    %% ---------- 汇总 ----------

    frame_snr = [];
    best_order = [];

    for k = 1:length(results.frames)
        frame_snr(end+1) = results.frames(k).snr_frame_rx1_db;
        best_order(end+1) = results.frames(k).best_tx_id;
    end

    valid_snr = frame_snr(isfinite(frame_snr));

    if isempty(valid_snr)
        file_snr_db = NaN;
    else
        file_snr_db = 10*log10(mean(10.^(valid_snr/10)));
    end

    fprintf('\n  --- %s Summary ---\n', mod_name);
    fprintf('  Extracted frames : %d / %d\n', length(results.frames), cfg.n_frames);
    fprintf('  Best TX order    : %s\n', mat2str(best_order));
    fprintf('  Frame SNR        : %s dB\n', mat2str(round(frame_snr, 2)));
    fprintf('  File SNR (avg)   : %.2f dB\n', file_snr_db);

    % 存储结果
    mod_result = struct();
    mod_result.mod_name = mod_name;
    mod_result.bits = bits;
    mod_result.M = Mq;
    mod_result.frame_snr = frame_snr;
    mod_result.file_snr_db = file_snr_db;
    mod_result.best_order = best_order;
    mod_result.n_frames_extracted = length(results.frames);
    mod_result.results = results;

    all_results{end+1} = mod_result;

end

%% ===================== 总汇总 =====================

fprintf('\n');
fprintf('============================================\n');
fprintf(' Overall Summary (sub03, strong turbulence)\n');
fprintf('============================================\n');
fprintf('%-8s | %6s | %12s | %10s | %s\n', ...
    'Mod', 'M', 'Frames', 'SNR(dB)', 'Constellation');
fprintf('---------+--------+--------------+------------+------------------\n');

for i = 1:length(all_results)
    r = all_results{i};
    n_frames = r.n_frames_extracted;
    snr_str = sprintf('%.2f', r.file_snr_db);

    % 星座检查
    const_check = '';
    switch r.mod_name
        case 'QPSK'
            const_check = '4 points';
        case '16QAM'
            const_check = 'expect 4x4';
        case '32QAM'
            const_check = 'expect cross';
        case '64QAM'
            const_check = 'expect 8x8';
        case '128QAM'
            const_check = 'expect cross';
        case '256QAM'
            const_check = 'expect 16x16';
    end

    fprintf('%-8s | %6d | %8d/%d  | %10s | %s\n', ...
        r.mod_name, r.M, n_frames, cfg.n_frames, snr_str, const_check);
end

fprintf('============================================\n');

%% ===================== 保存结果 =====================

out_dir = fullfile(cfg.data_root, 'uniformQAM_test_results');

if ~exist(out_dir, 'dir')
    mkdir(out_dir);
end

out_file = fullfile(out_dir, sprintf('rx_test_%s_bin%02d_%s.mat', ...
    cfg.rx_sub, cfg.test_bin, datestr(now, 'yyyy.mm.dd_HHMM')));

save(out_file, 'cfg', 'all_results', '-v7.3');

fprintf('\nResults saved to:\n  %s\n', out_file);
fprintf('\nDone!\n');