%% test_one_sample_rx1_snr.m
% 单样本测试：IM/DD 实值读取 + iterative RX1-style 同步解调 + rx1-style SNR估计
%
% 目的：
% 1. 对一个 .bin 文件提取 3 帧
% 2. 每帧手动 OFDM 解调得到 rx_sc
% 3. 每帧分别与 tx_frame1/2/3.mat 比较
% 4. 使用 rx1 中的 symDemapper + EVM_eval 方式计算 SNR
%
% 注意：
% - 不直接调用 deOFDM，避免 deOFDM 内部二次同步导致越界
% - .bin 按 IM/DD APD 直接探测的实值波形读取，不做 I+jQ 奇偶点组合
% - SNR 仅用于数据质量标注，不作为识别模型输入

clear; clear global; close all; clc;

addpath(genpath('D:\Project_code\OptDSP_FSO_2\practical waterfiling-exp\OptDSP_lite-master'));

global PROG;
PROG.showMessagesLevel = 1;
initProg();

%% ===================== 用户配置区 =====================

cfg = struct();

cfg.data_root = 'D:\Project_code\OptDSP_FSO_2\practical waterfiling-exp\OptDSP_lite-master\2_Data_Results';
cfg.rx_date   = '2026.06.29';
cfg.tx_root   = fullfile(cfg.data_root, 'tx_3frame_6mod_uniform_txt1');

% ====== 修改这里测试单个样本 ======
cfg.mod_name = '64QAM';
cfg.sub_name = 'sub2';
cfg.sig_idx  = 26;

% 如果要测试 32QAM/sub3/51：
% cfg.mod_name = '32QAM';
% cfg.sub_name = 'sub3';
% cfg.sig_idx  = 51;

cfg.Fs_rx   = 80e9;
cfg.Fs_base = 16e9;

cfg.n_target_frames = 3;

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

% rx1-style SNR 匹配时，允许少量 OFDM symbol shift
% 如果你确认完全对齐，可改为 0
cfg.shift_set = -5:5;

% 是否尝试 rx_sc 共轭
cfg.try_conjugate = true;

% 画图
cfg.plot_en = true;

%% ===================== 文件路径 =====================

rx_bin_file = fullfile(cfg.data_root, 'rx_data', cfg.rx_date, ...
    cfg.mod_name, cfg.sub_name, sprintf('%d.bin', cfg.sig_idx));

tx_frame_files = cell(1, cfg.n_target_frames);

for k = 1:cfg.n_target_frames
    tx_frame_files{k} = fullfile(cfg.tx_root, cfg.mod_name, cfg.sub_name, ...
        sprintf('sig_%04d_frame%d.mat', cfg.sig_idx, k));
end

fprintf('\n============================================\n');
fprintf(' Single Sample RX1-style SNR Test\n');
fprintf('============================================\n');
fprintf('RX bin : %s\n', rx_bin_file);
fprintf('Mod    : %s\n', cfg.mod_name);
fprintf('Sub    : %s\n', cfg.sub_name);
fprintf('SigIdx : %d\n', cfg.sig_idx);

%% ===================== OFDM 参数结构 =====================

ofdm = struct();
ofdm.NumberOfIFFTSamples = cfg.n_fft;
ofdm.NumberOfGuardTime   = cfg.n_guard;
ofdm.Carrier_location    = cfg.carrier_loc;
ofdm.Carrier_location_demo = [4:126, 132:254];
ofdm.NumberOfCarriers = length(ofdm.Carrier_location);
ofdm.NumberOfCarriers_demo = length(ofdm.Carrier_location_demo);
ofdm.size = cfg.n_syms;

%% ===================== QAM 参数 =====================

[Mq, bits] = mod_to_order_bits_local(cfg.mod_name);

nBpS_net = bits - 0.2 * (bits > 2);

TX.SIG = setSignalParams('symRate', 8e9, 'M', Mq, ...
    'nPol', 1, 'nBpS', nBpS_net, 'nSyms', cfg.n_syms, ...
    'roll-off', 0.25, 'modulation', 'QAM');

TX.QAM = QAM_config(TX.SIG);
C = TX.QAM.IQmap;

DSP = struct();
DSP.DEMAPPER.normMethod = 'MMSE';

%% ===================== 加载 TX 三帧参考 =====================

tx_refs = cell(1, cfg.n_target_frames);

for k = 1:cfg.n_target_frames

    if ~exist(tx_frame_files{k}, 'file')
        error('TX frame reference missing: %s', tx_frame_files{k});
    end

    tmp = load(tx_frame_files{k});

    if ~isfield(tmp, 'data_tx')
        error('No data_tx in reference file: %s', tx_frame_files{k});
    end

    tx_refs{k} = tmp.data_tx.';     % [123 × 128]

    fprintf('Loaded TX frame %d: %s\n', k, tx_frame_files{k});
end

%% ===================== 生成 LTS =====================

LTS = make_lts_local(cfg.n_fft);

%% ===================== 读取 RX .bin，注意：实值读取 =====================

fprintf('\n[1] Reading Keysight .bin as real IM/DD waveform...\n');

[rx80, read_info] = read_keysight_bin_robust_real_local(rx_bin_file);

rx80 = rx80(:).';
rx80 = rx80 - mean(rx80);
rx80 = rx80 ./ (rms(rx80) + eps);

fprintf('Read OK. RX80 length = %d samples\n', length(rx80));
fprintf('Reader method: %s, inferred bpp = %g\n', ...
    read_info.method, read_info.inferred_bpp);

%% ===================== Resample 80G → 16G =====================

fprintf('\n[2] Resampling %.0f GSa/s -> %.0f GSa/s...\n', ...
    cfg.Fs_rx/1e9, cfg.Fs_base/1e9);

rx16 = resample(rx80, cfg.Fs_base, cfg.Fs_rx);

rx16 = rx16(:).';
rx16 = rx16 - mean(rx16);
rx16 = rx16 ./ (rms(rx16) + eps);

fprintf('RX16 length = %d samples\n', length(rx16));

% AWG循环保护：末尾拼接开头一段，避免第三帧跨采样边界
wrap_len = min(length(rx16), 3 * cfg.frame_len_16);
rx16_ext = [rx16, rx16(1:wrap_len)];

fprintf('RX16 extended length = %d samples\n', length(rx16_ext));

%% ===================== Iterative RX1-style 单帧提取 + SNR =====================

cursor = 1;

results = struct();
results.frame = [];

fprintf('\n[3] Iterative RX1-style frame extraction and SNR estimation...\n');

for rk = 1:cfg.n_target_frames

    fprintf('\n--------------------------------------------\n');
    fprintf('Searching RX frame %d/%d\n', rk, cfg.n_target_frames);
    fprintf('Current cursor = %d\n', cursor);

    if cursor >= length(rx16_ext) - cfg.frame_len_16
        warning('Cursor too close to end. Stop.');
        break;
    end

    search_sig = rx16_ext(cursor:end);

    try
        [lts_start_rel, frame_start_rel, sync_info] = ...
            find_one_frame_start_rx1_style_local(search_sig, ofdm, cfg);

        lts_start_abs   = cursor + lts_start_rel - 1;
        frame_start_abs = cursor + frame_start_rel - 1;

        fprintf('Frame %d sync OK:\n', rk);
        fprintf('  edge_index      = %d\n', sync_info.edge_index);
        fprintf('  fine_time_est   = %d\n', sync_info.fine_time_est);
        fprintf('  lts_start_abs   = %d\n', lts_start_abs);
        fprintf('  frame_start_abs = %d\n', frame_start_abs);

    catch ME
        warning('Frame %d sync failed: %s', rk, ME.message);
        break;
    end

    try
        [rx_sc, rx_time, rx_frame16_lts, demod_info] = ...
            demod_one_frame_from_lts_start_local(rx16_ext, lts_start_abs, LTS, cfg);

        fprintf('Frame %d demod OK: n_use=%d, CFO=%.4e\n', ...
            rk, demod_info.n_use, demod_info.cfo);

    catch ME
        warning('Frame %d demod failed: %s', rk, ME.message);
        cursor = max(cursor + round(0.5 * cfg.frame_len_16), 1);
        continue;
    end

    try
        [best_tx_id, best_snr_db, best_snr_sc_db, ...
            snr_list_db, best_txafdem, best_ber_sc, align_info] = ...
            match_rx_to_tx_by_rx1_snr_local(rx_sc, tx_refs, C, DSP, cfg);

        fprintf('Frame %d RX1-style SNR OK:\n', rk);
        fprintf('  SNR list [tx1 tx2 tx3] = ');
        fprintf('%.2f ', snr_list_db);
        fprintf('dB\n');
        fprintf('  Best TX frame = %d\n', best_tx_id);
        fprintf('  Best SNR      = %.2f dB\n', best_snr_db);
        fprintf('  Align shift   = %d\n', align_info.shift);
        fprintf('  RX variant    = %s\n', align_info.rx_variant);

        valid_ber = best_ber_sc(isfinite(best_ber_sc));
        if ~isempty(valid_ber)
            fprintf('  BER mean      = %.4e\n', mean(valid_ber));
        else
            fprintf('  BER mean      = NaN\n');
        end

    catch ME
        warning('Frame %d RX1-style SNR failed: %s', rk, ME.message);
        best_tx_id = NaN;
        best_snr_db = NaN;
        best_snr_sc_db = NaN(cfg.n_sc,1);
        snr_list_db = NaN(1,3);
        best_txafdem = [];
        best_ber_sc = NaN(cfg.n_sc,1);
        align_info = struct();
    end

    % Power SQI，仅作诊断，不作为主 SNR
    power_sqi_db = 10 * log10(mean(abs(rx_sc(:)).^2) + eps);

    % 保存结果到结构体
    one = struct();
    one.rx_frame_idx = rk;
    one.rx_sc = rx_sc;
    one.rx_time = rx_time;
    one.rx_frame16_lts = rx_frame16_lts;
    one.best_tx_id = best_tx_id;
    one.snr_frame_rx1_db = best_snr_db;
    one.snr_sc_rx1_db = best_snr_sc_db;
    one.snr_list_db = snr_list_db;
    one.txafdem_matrix = best_txafdem;
    one.ber_sc = best_ber_sc;
    one.power_sqi_db = power_sqi_db;
    one.align_info = align_info;
    one.sync_info = sync_info;
    one.demod_info = demod_info;
    one.lts_start_abs = lts_start_abs;
    one.frame_start_abs = frame_start_abs;

    results.frame = [results.frame, one];

    % 画星座图 + 每个子载波的 SNR 曲线
    if cfg.plot_en
        figure('Name', sprintf('RX frame %d constellation', rk));
        subplot(1,2,1);
        plot(rx_sc(:), 'b.');
        axis square;
        grid on;
        title(sprintf('RX frame %d equalized rx\\_sc', rk));
        xlabel('I');
        ylabel('Q');

        subplot(1,2,2);
        if ~isempty(best_txafdem)
            plot(best_txafdem(:), 'r.');
            axis square;
            grid on;
            title(sprintf('Decision symbols, SNR=%.2f dB', best_snr_db));
            xlabel('I');
            ylabel('Q');
        else
            text(0.1,0.5,'No txafdem');
            axis off;
        end

        % 每个子载波的 SNR 曲线
        figure('Name', sprintf('RX frame %d SNR per subcarrier', rk));
        plot(best_snr_sc_db, 'b.-');
        grid on;
        xlabel('Subcarrier index');
        ylabel('SNR (dB)');
        title(sprintf('Frame %d SNR per subcarrier (avg=%.2f dB)', rk, best_snr_db));
    end

    % 推进 cursor：从当前完整帧后方略微提前一点开始找下一帧
    cursor_next = frame_start_abs + cfg.frame_len_16 - cfg.next_search_backoff;

    if cursor_next <= cursor
        cursor_next = cursor + round(0.8 * cfg.frame_len_16);
    end

    cursor = cursor_next;
end

%% ===================== 文件级 SNR =====================

frame_snr = [];

for k = 1:length(results.frame)
    s = results.frame(k).snr_frame_rx1_db;
    if isfinite(s)
        frame_snr(end+1) = s;
    end
end

if ~isempty(frame_snr)
    file_snr_rx1_db = 10 * log10(mean(10.^(frame_snr/10)));
else
    file_snr_rx1_db = NaN;
end

fprintf('\n============================================\n');
fprintf(' Test Summary\n');
fprintf('============================================\n');
fprintf('Extracted frames: %d/%d\n', length(results.frame), cfg.n_target_frames);
fprintf('Frame RX1-style SNR: ');
fprintf('%.2f ', frame_snr);
fprintf('dB\n');
fprintf('File RX1-style SNR = %.2f dB\n', file_snr_rx1_db);

fprintf('Best TX order: ');
for k = 1:length(results.frame)
    fprintf('%d ', results.frame(k).best_tx_id);
end
fprintf('\n');

fprintf('============================================\n');

%% ===================== 可选保存 =====================

out_dir = fullfile(cfg.data_root, 'single_sample_rx1_snr_test');

if ~exist(out_dir, 'dir')
    mkdir(out_dir);
end

out_file = fullfile(out_dir, sprintf('%s_%s_sig%04d_rx1snr_test.mat', ...
    cfg.mod_name, cfg.sub_name, cfg.sig_idx));

save(out_file, 'cfg', 'results', 'file_snr_rx1_db', '-v7.3');

fprintf('Saved test result to:\n%s\n', out_file);

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