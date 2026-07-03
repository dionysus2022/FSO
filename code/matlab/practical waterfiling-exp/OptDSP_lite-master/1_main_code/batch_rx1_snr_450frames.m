%% batch_rx1_snr_450frames.m
% 批处理版本：IM/DD 实值读取 + iterative RX1-style 同步解调 + rx1-style SNR估计
%
% 使用场景：
% 1. 第一次采集数据：sub01 = 弱湍流，sub03 = 强湍流。
% 2. 每种湍流按 150 个 .bin 文件 × 每个 .bin 提取 3 帧 = 450 帧信号。
% 3. 关闭所有画图，批量保存每个 .bin 的 3 帧处理结果，并生成总索引 summary。
%
% 重要说明：
% - 本脚本保留你原来 test_one_sample_rx1_snr.m 中已经验证正确的读取、同步、OFDM 解调、SNR 匹配核心逻辑。
% - 不直接调用 deOFDM，避免 deOFDM 内部二次同步导致越界。
% - .bin 按 IM/DD APD 直接探测的实值波形读取，不做 I+jQ 奇偶点组合。
% - SNR 仅用于数据质量标注，不作为识别模型输入。
% - 默认处理 6 种调制格式；如果只想处理某一种，把 cfg.mod_list 改成 {'64QAM'} 这类单元素即可。

clear; clear global; close all; clc;

%% ===================== 工程路径 =====================

addpath(genpath('D:\Project_code\OptDSP_FSO_2\practical waterfiling-exp\OptDSP_lite-master'));

global PROG;
PROG.showMessagesLevel = 1;
initProg();

% 批处理时彻底关闭图窗，避免 450 帧循环时卡死
set(0, 'DefaultFigureVisible', 'off');

%% ===================== 用户配置区 =====================

cfg = struct();

cfg.data_root = 'D:\Project_code\OptDSP_FSO_2\practical waterfiling-exp\OptDSP_lite-master\2_Data_Results';
cfg.rx_date   = '2026.06.26';
cfg.tx_root   = fullfile(cfg.data_root, 'tx_3frame_6mod');

% 输出目录：每个 .bin 一个 .mat，另外保存总 summary
cfg.out_root  = fullfile(cfg.data_root, 'rx1_batch_450frames_results', cfg.rx_date);

% 第一次采集数据：sub01 弱湍流，sub03 强湍流
cfg.turbulence_list = struct( ...
    'name',     {'weak',  'strong'}, ...
    'cn_name',  {'弱湍流', '强湍流'}, ...
    'sub_name', {'sub01', 'sub03'});

% 默认处理 6 种调制。若只想跑单一调制，例如 64QAM：
% cfg.mod_list = {'64QAM'};
cfg.mod_list = {'QPSK', '16QAM', '32QAM', '64QAM', '128QAM', '256QAM'};

% 当前单样本脚本是一份 .bin 提取 3 帧。
% 因此 150 个 .bin × 3 帧 = 450 帧。
cfg.sig_idx_list = 1:150;
cfg.n_target_frames = 3;

% 如果你的真实文件夹里不是 150 个 .bin，而是 450 个 .bin 且每个 .bin 只对应 1 帧，改成：
% cfg.sig_idx_list = 1:450;
% cfg.n_target_frames = 1;

cfg.Fs_rx   = 80e9;
cfg.Fs_base = 16e9;

cfg.zeros_head = 80;
cfg.n_fft      = 256;
cfg.n_guard    = 16;
cfg.n_syms     = 128;

cfg.carrier_loc = 4:126;
cfg.n_sc = length(cfg.carrier_loc);
cfg.sym_len = cfg.n_fft + cfg.n_guard;

cfg.frame_pre_lts = cfg.zeros_head + cfg.n_guard - 5;   % 与 deOFDM 中 fine_index 前退逻辑一致
cfg.header_len_16 = cfg.zeros_head + cfg.n_guard + 2*cfg.n_fft;
cfg.frame_len_16  = cfg.header_len_16 + cfg.sym_len * cfg.n_syms;

cfg.M_time = 32768;

% 给 iterative 搜索用，避免下一次搜索还落在当前帧内部
cfg.next_search_backoff = 2000;

% rx1-style SNR 匹配时，允许少量 OFDM symbol shift。
% 如果你确认完全对齐，可改为 0。
cfg.shift_set = -5:5;

% 是否尝试 rx_sc 共轭
cfg.try_conjugate = true;

% 批处理必须关闭画图
cfg.plot_en = false;

% 保存控制：为了后续 GNN / 识别数据集，建议保留 rx_sc 和 rx_time。
% 如果磁盘压力很大，可以关闭 rx_frame16_lts 或 txafdem_matrix。
cfg.save_rx_sc          = true;
cfg.save_rx_time        = true;
cfg.save_rx_frame16_lts = true;
cfg.save_txafdem_matrix = false;  % 批量时该矩阵通常较占空间，默认不保存；需要诊断时改 true
cfg.save_as_single      = true;   % 将大数组转 single，明显节省磁盘

% 缺失文件是否打印 warning。批量跑时建议 false，避免刷屏。
cfg.verbose_missing = false;

%% ===================== OFDM 参数结构 =====================

ofdm = struct();
ofdm.NumberOfIFFTSamples = cfg.n_fft;
ofdm.NumberOfGuardTime   = cfg.n_guard;
ofdm.Carrier_location    = cfg.carrier_loc;
ofdm.Carrier_location_demo = [4:126, 132:254];
ofdm.NumberOfCarriers = length(ofdm.Carrier_location);
ofdm.NumberOfCarriers_demo = length(ofdm.Carrier_location_demo);
ofdm.size = cfg.n_syms;

%% ===================== 生成 LTS =====================

LTS = make_lts_local(cfg.n_fft);

%% ===================== 输出目录与日志 =====================

if ~exist(cfg.out_root, 'dir')
    mkdir(cfg.out_root);
end

log_file = fullfile(cfg.out_root, sprintf('batch_log_%s.txt', datestr(now, 'yyyymmdd_HHMMSS')));
diary(log_file);
diary on;

fprintf('\n============================================================\n');
fprintf(' Batch RX1-style Processing: 450 frames per turbulence\n');
fprintf('============================================================\n');
fprintf('RX root    : %s\n', fullfile(cfg.data_root, 'rx_data', cfg.rx_date));
fprintf('TX root    : %s\n', cfg.tx_root);
fprintf('Output root: %s\n', cfg.out_root);
fprintf('Sub01      : weak turbulence / 弱湍流\n');
fprintf('Sub03      : strong turbulence / 强湍流\n');
fprintf('Sig idx    : %d to %d, total %d .bin per mod/sub\n', ...
    cfg.sig_idx_list(1), cfg.sig_idx_list(end), numel(cfg.sig_idx_list));
fprintf('Frames/bin : %d\n', cfg.n_target_frames);
fprintf('Expected   : %d frames per mod/sub\n', numel(cfg.sig_idx_list) * cfg.n_target_frames);
fprintf('Plot       : OFF\n');
fprintf('============================================================\n');

%% ===================== 批处理主循环 =====================

summary_rows = struct([]);
missing_rx = {};
missing_tx = {};
failed_files = struct([]);
processed_file_count = 0;
processed_frame_count = 0;

batch_tic = tic;

total_jobs = numel(cfg.turbulence_list) * numel(cfg.mod_list) * numel(cfg.sig_idx_list);
job_id = 0;

for ti = 1:numel(cfg.turbulence_list)

    turb = cfg.turbulence_list(ti);

    fprintf('\n############################################################\n');
    fprintf('Turbulence: %s / %s / folder=%s\n', turb.name, turb.cn_name, turb.sub_name);
    fprintf('############################################################\n');

    for mi = 1:numel(cfg.mod_list)

        mod_name = cfg.mod_list{mi};

        fprintf('\n==================== Modulation: %s ====================\n', mod_name);

        for si = 1:numel(cfg.sig_idx_list)

            sig_idx = cfg.sig_idx_list(si);
            job_id = job_id + 1;

            fprintf('\n[%d/%d] %s | %s | sig_idx=%04d\n', ...
                job_id, total_jobs, turb.cn_name, mod_name, sig_idx);

            [rx_bin_file, rx_found] = find_rx_bin_file_local(cfg, mod_name, turb.sub_name, sig_idx);

            if ~rx_found
                missing_rx{end+1,1} = sprintf('%s | %s | sig_%04d', turb.sub_name, mod_name, sig_idx); %#ok<SAGROW>
                if cfg.verbose_missing
                    warning('RX bin missing: %s | %s | sig_%04d', turb.sub_name, mod_name, sig_idx);
                end
                continue;
            end

            [tx_frame_files, tx_found, tx_missing_list] = find_tx_frame_files_local(cfg, mod_name, turb.sub_name, sig_idx);

            if ~tx_found
                missing_tx{end+1,1} = sprintf('%s | %s | sig_%04d | missing=%s', ...
                    turb.sub_name, mod_name, sig_idx, strjoin(tx_missing_list, ', ')); %#ok<SAGROW>
                if cfg.verbose_missing
                    warning('TX reference missing: %s | %s | sig_%04d', turb.sub_name, mod_name, sig_idx);
                end
                continue;
            end

            try
                [results, file_snr_rx1_db, file_status] = process_one_rx_bin_local( ...
                    rx_bin_file, tx_frame_files, cfg, ofdm, LTS, mod_name, turb, sig_idx);

                out_dir = fullfile(cfg.out_root, sprintf('%s_%s', turb.name, turb.sub_name), mod_name);
                if ~exist(out_dir, 'dir')
                    mkdir(out_dir);
                end

                out_file = fullfile(out_dir, sprintf('%s_%s_sig%04d_rx1_batch.mat', ...
                    mod_name, turb.sub_name, sig_idx));

                cfg_save = cfg;
                save(out_file, 'cfg_save', 'mod_name', 'turb', 'sig_idx', ...
                    'rx_bin_file', 'tx_frame_files', 'results', 'file_snr_rx1_db', 'file_status', '-v7.3');

                processed_file_count = processed_file_count + 1;
                processed_frame_count = processed_frame_count + numel(results.frame);

                summary_rows = append_summary_rows_local(summary_rows, ...
                    results, file_snr_rx1_db, file_status, out_file, rx_bin_file, mod_name, turb, sig_idx); %#ok<AGROW>

                fprintf('Saved: %s\n', out_file);
                fprintf('Extracted frames: %d/%d, file SNR = %.2f dB\n', ...
                    numel(results.frame), cfg.n_target_frames, file_snr_rx1_db);

            catch ME
                warn_msg = sprintf('FAILED file: %s | error: %s', rx_bin_file, ME.message);
                warning('%s', warn_msg);

                ff = struct();
                ff.turbulence = turb.name;
                ff.sub_name = turb.sub_name;
                ff.mod_name = mod_name;
                ff.sig_idx = sig_idx;
                ff.rx_bin_file = rx_bin_file;
                ff.error_msg = ME.message;
                failed_files = [failed_files; ff]; %#ok<AGROW>
            end
        end
    end
end

%% ===================== 保存总索引 =====================

summary_file_mat = fullfile(cfg.out_root, 'batch_summary.mat');
summary_file_csv = fullfile(cfg.out_root, 'batch_summary.csv');
missing_file_mat = fullfile(cfg.out_root, 'batch_missing_and_failed.mat');

if isempty(summary_rows)
    summary_table = table();
else
    summary_table = struct2table(summary_rows);
end

save(summary_file_mat, 'cfg', 'summary_rows', 'summary_table', '-v7.3');

try
    writetable(summary_table, summary_file_csv);
catch ME
    warning('Failed to write summary csv: %s', ME.message);
end

save(missing_file_mat, 'missing_rx', 'missing_tx', 'failed_files', '-v7.3');

elapsed_sec = toc(batch_tic);

fprintf('\n============================================================\n');
fprintf(' Batch finished.\n');
fprintf('Processed files : %d\n', processed_file_count);
fprintf('Processed frames: %d\n', processed_frame_count);
fprintf('Missing RX      : %d\n', numel(missing_rx));
fprintf('Missing TX      : %d\n', numel(missing_tx));
fprintf('Failed files    : %d\n', numel(failed_files));
fprintf('Elapsed         : %.2f sec = %.2f min\n', elapsed_sec, elapsed_sec/60);
fprintf('Summary MAT     : %s\n', summary_file_mat);
fprintf('Summary CSV     : %s\n', summary_file_csv);
fprintf('Missing/Failed  : %s\n', missing_file_mat);
fprintf('Log file        : %s\n', log_file);
fprintf('============================================================\n');

diary off;

%% =====================================================================
%% Batch Helper Functions
%% =====================================================================

function [results, file_snr_rx1_db, file_status] = process_one_rx_bin_local( ...
    rx_bin_file, tx_frame_files, cfg, ofdm, LTS, mod_name, turb, sig_idx)

    file_status = struct();
    file_status.ok = false;
    file_status.error_msg = '';
    file_status.mod_name = mod_name;
    file_status.turbulence = turb.name;
    file_status.turbulence_cn = turb.cn_name;
    file_status.sub_name = turb.sub_name;
    file_status.sig_idx = sig_idx;
    file_status.rx_bin_file = rx_bin_file;

    %% QAM 参数
    [Mq, bits] = mod_to_order_bits_local(mod_name);

    nBpS_net = bits - 0.2 * (bits > 2);

    TX = struct();
    TX.SIG = setSignalParams('symRate', 8e9, 'M', Mq, ...
        'nPol', 1, 'nBpS', nBpS_net, 'nSyms', cfg.n_syms, ...
        'roll-off', 0.25, 'modulation', 'QAM');

    TX.QAM = QAM_config(TX.SIG);
    C = TX.QAM.IQmap;

    DSP = struct();
    DSP.DEMAPPER.normMethod = 'MMSE';

    %% 加载 TX 三帧参考
    tx_refs = cell(1, cfg.n_target_frames);

    for k = 1:cfg.n_target_frames
        tmp = load(tx_frame_files{k});

        if ~isfield(tmp, 'data_tx')
            error('No data_tx in reference file: %s', tx_frame_files{k});
        end

        tx_refs{k} = tmp.data_tx.';     % [123 × 128]
    end

    %% 读取 RX .bin，注意：实值读取
    [rx80, read_info] = read_keysight_bin_robust_real_local(rx_bin_file);

    rx80 = rx80(:).';
    rx80 = rx80 - mean(rx80);
    rx80 = rx80 ./ (rms(rx80) + eps);

    %% Resample 80G → 16G
    rx16 = resample(rx80, cfg.Fs_base, cfg.Fs_rx);

    rx16 = rx16(:).';
    rx16 = rx16 - mean(rx16);
    rx16 = rx16 ./ (rms(rx16) + eps);

    % AWG循环保护：末尾拼接开头一段，避免第三帧跨采样边界
    wrap_len = min(length(rx16), 3 * cfg.frame_len_16);
    rx16_ext = [rx16, rx16(1:wrap_len)];

    %% Iterative RX1-style 单帧提取 + SNR
    cursor = 1;

    results = struct();
    results.frame = [];
    results.read_info = read_info;
    results.rx80_len = length(rx80);
    results.rx16_len = length(rx16);
    results.rx16_ext_len = length(rx16_ext);

    for rk = 1:cfg.n_target_frames

        if cursor >= length(rx16_ext) - cfg.frame_len_16
            warning('Cursor too close to end. Stop. file=%s, rk=%d', rx_bin_file, rk);
            break;
        end

        search_sig = rx16_ext(cursor:end);

        try
            [lts_start_rel, frame_start_rel, sync_info] = ...
                find_one_frame_start_rx1_style_local(search_sig, ofdm, cfg);

            lts_start_abs   = cursor + lts_start_rel - 1;
            frame_start_abs = cursor + frame_start_rel - 1;

        catch ME
            warning('Frame sync failed. file=%s, rk=%d, error=%s', rx_bin_file, rk, ME.message);
            break;
        end

        try
            [rx_sc, rx_time, rx_frame16_lts, demod_info] = ...
                demod_one_frame_from_lts_start_local(rx16_ext, lts_start_abs, LTS, cfg);

        catch ME
            warning('Frame demod failed. file=%s, rk=%d, error=%s', rx_bin_file, rk, ME.message);
            cursor = max(cursor + round(0.5 * cfg.frame_len_16), 1);
            continue;
        end

        try
            [best_tx_id, best_snr_db, best_snr_sc_db, ...
                snr_list_db, best_txafdem, best_ber_sc, align_info] = ...
                match_rx_to_tx_by_rx1_snr_local(rx_sc, tx_refs, C, DSP, cfg);

        catch ME
            warning('RX1-style SNR failed. file=%s, rk=%d, error=%s', rx_bin_file, rk, ME.message);
            best_tx_id = NaN;
            best_snr_db = NaN;
            best_snr_sc_db = NaN(cfg.n_sc,1);
            snr_list_db = NaN(1,cfg.n_target_frames);
            best_txafdem = [];
            best_ber_sc = NaN(cfg.n_sc,1);
            align_info = struct();
        end

        % Power SQI，仅作诊断，不作为主 SNR
        power_sqi_db = 10 * log10(mean(abs(rx_sc(:)).^2) + eps);

        % 批量保存时压缩大数组
        if cfg.save_as_single
            rx_sc_save = single(rx_sc);
            rx_time_save = single(rx_time);
            rx_frame16_lts_save = single(rx_frame16_lts);
            best_snr_sc_db_save = single(best_snr_sc_db);
            best_ber_sc_save = single(best_ber_sc);
            if ~isempty(best_txafdem)
                best_txafdem_save = single(best_txafdem);
            else
                best_txafdem_save = [];
            end
        else
            rx_sc_save = rx_sc;
            rx_time_save = rx_time;
            rx_frame16_lts_save = rx_frame16_lts;
            best_snr_sc_db_save = best_snr_sc_db;
            best_ber_sc_save = best_ber_sc;
            best_txafdem_save = best_txafdem;
        end

        one = struct();
        one.turbulence = turb.name;
        one.turbulence_cn = turb.cn_name;
        one.sub_name = turb.sub_name;
        one.mod_name = mod_name;
        one.sig_idx = sig_idx;
        one.rx_frame_idx = rk;                 % 当前 .bin 内第几帧：1/2/3
        one.global_frame_idx = (sig_idx - cfg.sig_idx_list(1)) * cfg.n_target_frames + rk;  % 当前湍流/调制下全局帧号

        if cfg.save_rx_sc
            one.rx_sc = rx_sc_save;
        else
            one.rx_sc = [];
        end

        if cfg.save_rx_time
            one.rx_time = rx_time_save;
        else
            one.rx_time = [];
        end

        if cfg.save_rx_frame16_lts
            one.rx_frame16_lts = rx_frame16_lts_save;
        else
            one.rx_frame16_lts = [];
        end

        one.best_tx_id = best_tx_id;
        one.snr_frame_rx1_db = best_snr_db;
        one.snr_sc_rx1_db = best_snr_sc_db_save;
        one.snr_list_db = snr_list_db;

        if cfg.save_txafdem_matrix
            one.txafdem_matrix = best_txafdem_save;
        else
            one.txafdem_matrix = [];
        end

        one.ber_sc = best_ber_sc_save;
        one.power_sqi_db = power_sqi_db;
        one.align_info = align_info;
        one.sync_info = sync_info;
        one.demod_info = demod_info;
        one.lts_start_abs = lts_start_abs;
        one.frame_start_abs = frame_start_abs;
        one.cursor_start = cursor;

        results.frame = [results.frame, one]; %#ok<AGROW>

        % 推进 cursor：从当前完整帧后方略微提前一点开始找下一帧
        cursor_next = frame_start_abs + cfg.frame_len_16 - cfg.next_search_backoff;

        if cursor_next <= cursor
            cursor_next = cursor + round(0.8 * cfg.frame_len_16);
        end

        cursor = cursor_next;
    end

    %% 文件级 SNR
    frame_snr = [];

    for k = 1:length(results.frame)
        s = results.frame(k).snr_frame_rx1_db;
        if isfinite(s)
            frame_snr(end+1) = s; %#ok<AGROW>
        end
    end

    if ~isempty(frame_snr)
        file_snr_rx1_db = 10 * log10(mean(10.^(frame_snr/10)));
    else
        file_snr_rx1_db = NaN;
    end

    file_status.ok = true;
    file_status.n_extracted_frames = numel(results.frame);
    file_status.n_target_frames = cfg.n_target_frames;
    file_status.frame_snr = frame_snr;
    file_status.file_snr_rx1_db = file_snr_rx1_db;
end

function [rx_bin_file, found] = find_rx_bin_file_local(cfg, mod_name, sub_name, sig_idx)

    found = false;
    rx_bin_file = '';

    rx_root = fullfile(cfg.data_root, 'rx_data', cfg.rx_date);
    sub_candidates = sub_name_candidates_local(sub_name);
    bin_names = bin_name_candidates_local(sig_idx);

    candidates = {};

    for si = 1:numel(sub_candidates)
        sub_i = sub_candidates{si};
        for bi = 1:numel(bin_names)
            bname = bin_names{bi};

            % 兼容原单样本脚本布局：rx_data/date/mod/sub/51.bin
            candidates{end+1} = fullfile(rx_root, mod_name, sub_i, bname); %#ok<AGROW>

            % 兼容用户描述布局：rx_data/date/sub/mod/51.bin
            candidates{end+1} = fullfile(rx_root, sub_i, mod_name, bname); %#ok<AGROW>

            % 如果子文件夹下直接放 bin，同时文件名带调制名，也尝试兼容
            candidates{end+1} = fullfile(rx_root, sub_i, sprintf('%s_%s', mod_name, bname)); %#ok<AGROW>
            candidates{end+1} = fullfile(rx_root, mod_name, sprintf('%s_%s', sub_i, bname)); %#ok<AGROW>
        end
    end

    for ci = 1:numel(candidates)
        if exist(candidates{ci}, 'file')
            rx_bin_file = candidates{ci};
            found = true;
            return;
        end
    end
end

function [tx_frame_files, found, missing_list] = find_tx_frame_files_local(cfg, mod_name, sub_name, sig_idx)

    tx_frame_files = cell(1, cfg.n_target_frames);
    found = true;
    missing_list = {};

    tx_root_mod = fullfile(cfg.tx_root, mod_name);
    sub_candidates = sub_name_candidates_local(sub_name);

    for k = 1:cfg.n_target_frames

        frame_candidates = {};

        for si = 1:numel(sub_candidates)
            sub_i = sub_candidates{si};
            frame_candidates{end+1} = fullfile(tx_root_mod, sub_i, sprintf('sig_%04d_frame%d.mat', sig_idx, k)); %#ok<AGROW>
            frame_candidates{end+1} = fullfile(tx_root_mod, sub_i, sprintf('sig_%d_frame%d.mat', sig_idx, k)); %#ok<AGROW>
            frame_candidates{end+1} = fullfile(cfg.tx_root, sub_i, mod_name, sprintf('sig_%04d_frame%d.mat', sig_idx, k)); %#ok<AGROW>
            frame_candidates{end+1} = fullfile(cfg.tx_root, sub_i, mod_name, sprintf('sig_%d_frame%d.mat', sig_idx, k)); %#ok<AGROW>
        end

        this_found = false;
        for ci = 1:numel(frame_candidates)
            if exist(frame_candidates{ci}, 'file')
                tx_frame_files{k} = frame_candidates{ci};
                this_found = true;
                break;
            end
        end

        if ~this_found
            found = false;
            missing_list{end+1} = sprintf('frame%d', k); %#ok<AGROW>
        end
    end
end

function names = sub_name_candidates_local(sub_name)

    names = {sub_name};

    token = regexp(sub_name, '^sub0*([0-9]+)$', 'tokens', 'once');
    if ~isempty(token)
        n = str2double(token{1});
        names{end+1} = sprintf('sub%d', n); %#ok<AGROW>
        names{end+1} = sprintf('sub%02d', n); %#ok<AGROW>
        names{end+1} = sprintf('sub%03d', n); %#ok<AGROW>
    end

    names = unique(names, 'stable');
end

function names = bin_name_candidates_local(sig_idx)

    names = {sprintf('%d.bin', sig_idx), sprintf('%04d.bin', sig_idx), sprintf('sig_%04d.bin', sig_idx), sprintf('sig_%d.bin', sig_idx)};
    names = unique(names, 'stable');
end

function summary_rows = append_summary_rows_local(summary_rows, ...
    results, file_snr_rx1_db, file_status, out_file, rx_bin_file, mod_name, turb, sig_idx)

    for k = 1:numel(results.frame)
        fr = results.frame(k);

        row = struct();
        row.turbulence = string(turb.name);
        row.turbulence_cn = string(turb.cn_name);
        row.sub_name = string(turb.sub_name);
        row.mod_name = string(mod_name);
        row.sig_idx = sig_idx;
        row.rx_frame_idx = fr.rx_frame_idx;
        row.global_frame_idx = fr.global_frame_idx;
        row.ok = true;
        row.best_tx_id = fr.best_tx_id;
        row.snr_frame_rx1_db = fr.snr_frame_rx1_db;
        row.file_snr_rx1_db = file_snr_rx1_db;
        row.power_sqi_db = fr.power_sqi_db;
        row.align_shift = safe_nested_field_local(fr, {'align_info', 'shift'}, NaN);
        row.rx_variant = string(safe_nested_field_local(fr, {'align_info', 'rx_variant'}, ''));
        row.lts_start_abs = fr.lts_start_abs;
        row.frame_start_abs = fr.frame_start_abs;
        row.cfo = safe_nested_field_local(fr, {'demod_info', 'cfo'}, NaN);
        row.n_use = safe_nested_field_local(fr, {'demod_info', 'n_use'}, NaN);
        row.n_extracted_frames_in_file = file_status.n_extracted_frames;
        row.rx_bin_file = string(rx_bin_file);
        row.out_file = string(out_file);

        if isempty(summary_rows)
            summary_rows = row;
        else
            summary_rows(end+1,1) = row; %#ok<AGROW>
        end
    end
end

function v = safe_nested_field_local(s, field_chain, default_value)

    v = default_value;
    tmp = s;

    for i = 1:numel(field_chain)
        f = field_chain{i};
        if isstruct(tmp) && isfield(tmp, f)
            tmp = tmp.(f);
        else
            return;
        end
    end

    if ~isempty(tmp)
        v = tmp;
    end
end


%% =====================================================================
%% Helper Functions
%% =====================================================================

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

function LTS = make_lts_local(n_fft)

    load('LongTrainSym_ini.mat', 'LongTrainSym_ini');

    LTS_f0 = LongTrainSym_ini(1:n_fft);
    LTS_f0([1 n_fft/2+1]) = 0;

    ltrs_in = LTS_f0;
    ltrs_in(1, n_fft/2+2:n_fft) = conj(ltrs_in(1, n_fft/2:-1:2));

    LTS.freq = ltrs_in(:);
    LTS.time = ifft(ltrs_in(:));
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

            % IM/DD APD direct-detection: single real waveform.
            % Do NOT convert odd/even samples into I+jQ.
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

%% ===================== RX1-style sync =====================

function [lts_start, frame_start, info] = ...
    find_one_frame_start_rx1_style_local(rx, ofdm, cfg)

    rx = rx(:).';

    n_fft = ofdm.NumberOfIFFTSamples;
    n_guard = ofdm.NumberOfGuardTime;
    n_syms = cfg.n_syms;

    symbol_bits = cfg.zeros_head + n_guard + 2*n_fft + ...
        (n_fft + n_guard) * n_syms;

    search_len = min(length(rx), 2 * symbol_bits);

    if search_len < symbol_bits
        error('input too short for one-frame sync: len=%d', length(rx));
    end

    search_sig = rx(1:search_len);

    [detected_packet, edge_index] = ...
        packet_edge_power_dect(search_sig, cfg.zeros_head);

    load('LongTrainSym_ini.mat', 'LongTrainSym_ini');

    LTS_f = LongTrainSym_ini(1:n_fft);
    LTS_f([1 n_fft/2+1]) = 0;

    ltrs_in = LTS_f;
    ltrs_in(1, n_fft/2+2:n_fft) = conj(ltrs_in(1, n_fft/2:-1:2));

    [fine_time_est, data_df, max_peak_long] = ...
        rx_fine_time_sync_cross_corr( ...
        detected_packet, n_guard, ltrs_in, cfg.zeros_head, 0);

    % 这个位置是第一个 LTS 的开始位置
    lts_start = edge_index + fine_time_est - 1;

    % 完整帧头位置，尽量和 deOFDM 的窗口定义保持一致
    frame_start = lts_start - cfg.frame_pre_lts;

    if frame_start < 1
        frame_start = 1;
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

%% ===================== Manual OFDM demod from LTS =====================

function [rx_sc, rx_time, rx_frame16_lts, info] = ...
    demod_one_frame_from_lts_start_local(rx, lts_start, LTS, cfg)

    rx = rx(:).';

    n_fft = cfg.n_fft;
    n_guard = cfg.n_guard;
    sym_len = cfg.sym_len;

    if lts_start < 1
        error('invalid lts_start=%d', lts_start);
    end

    lts1_start = lts_start;
    lts1_end   = lts_start + n_fft - 1;
    lts2_start = lts_start + n_fft;
    lts2_end   = lts_start + 2*n_fft - 1;

    if lts2_end > length(rx)
        error('frame too short for LTS: remaining=%d', length(rx)-lts_start+1);
    end

    lts1 = rx(lts1_start:lts1_end);
    lts2 = rx(lts2_start:lts2_end);

    cfo = angle(sum(lts1(:).*conj(lts2(:)))) / (2*pi*n_fft);

    remaining = length(rx) - lts_start + 1;
    n = 0:remaining-1;

    rx_comp = rx(lts_start:end) .* exp(-1j*2*pi*cfo*n/n_fft);

    lts1c = rx_comp(1:n_fft);
    lts2c = rx_comp(n_fft+1:2*n_fft);

    data_start = 2*n_fft + 1;
    dp_all = rx_comp(data_start:end);

    nd = floor(length(dp_all) / sym_len);

    if nd <= 0
        error('no complete OFDM symbols: nd=0');
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

    n_save = min(length(rx_comp), 2*n_fft + cfg.n_syms*sym_len);
    rx_frame16_lts = rx_comp(1:n_save);

    info = struct();
    info.lts_start = lts_start;
    info.cfo = cfo;
    info.n_use = n_use;
    info.nd_available = nd;
    info.data_start = data_start;
    info.remaining = remaining;
end

%% ===================== RX1-style SNR matching =====================

function [best_tx_id, best_snr_db, best_snr_sc_db, ...
    snr_list_db, best_txafdem, best_ber_sc, align_info] = ...
    match_rx_to_tx_by_rx1_snr_local(rx_sc, tx_refs, C, DSP_template, cfg)

    n_ref = length(tx_refs);

    snr_list_db = NaN(1, n_ref);
    snr_sc_list = cell(1, n_ref);
    txafdem_list = cell(1, n_ref);
    ber_list = cell(1, n_ref);
    info_list = cell(1, n_ref);

    best_global_snr = -inf;
    best_tx_id = NaN;
    best_snr_sc_db = NaN(size(rx_sc,1),1);
    best_txafdem = [];
    best_ber_sc = NaN(size(rx_sc,1),1);
    align_info = struct();

    for tid = 1:n_ref

        tx_ref0 = normalize_tx_ref_shape_for_rx1_local(tx_refs{tid}, size(rx_sc,1));

        best_tid_snr = -inf;
        best_tid_snr_sc = NaN(size(rx_sc,1),1);
        best_tid_txafdem = [];
        best_tid_ber = NaN(size(rx_sc,1),1);
        best_tid_info = struct();

        if cfg.try_conjugate
            rx_variants = {rx_sc, conj(rx_sc)};
            rx_names = {'normal', 'conj'};
        else
            rx_variants = {rx_sc};
            rx_names = {'normal'};
        end

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
                        best_tid_snr_sc = snr_sc_db;
                        best_tid_txafdem = txafdem_matrix;
                        best_tid_ber = ber_sc;

                        best_tid_info.tx_id = tid;
                        best_tid_info.shift = sh;
                        best_tid_info.rx_variant = rx_names{rv};
                        best_tid_info.n_sc = size(rx_use,1);
                        best_tid_info.n_sym = size(rx_use,2);
                    end

                catch
                    continue;
                end
            end
        end

        snr_list_db(tid) = best_tid_snr;
        snr_sc_list{tid} = best_tid_snr_sc;
        txafdem_list{tid} = best_tid_txafdem;
        ber_list{tid} = best_tid_ber;
        info_list{tid} = best_tid_info;

        if isfinite(best_tid_snr) && best_tid_snr > best_global_snr

            best_global_snr = best_tid_snr;
            best_tx_id = tid;
            best_snr_sc_db = best_tid_snr_sc;
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
        align_info = struct();
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
        [~, snr_tmp] = EVM_eval( ...
            rx_use(valid_rows,:), ...
            txafdem_matrix(valid_rows,:));

        snr_tmp = snr_tmp(:);

        n_fill = min(length(valid_rows), length(snr_tmp));
        snr_sc_db(valid_rows(1:n_fill)) = snr_tmp(1:n_fill);

    catch
        % 如果整体 EVM_eval 失败，则逐子载波尝试
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