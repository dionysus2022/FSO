%% iterative_rx1_style_pipeline.m
% 使用 rx1 单帧同步逻辑，递归识别/切分 3 帧信号
% 修改3：不再调用 deOFDM，从 LTS 起点直接解调
% 修改4：集成三层 SNR 结构（snr分析流程.txt）

clear; clear global; close all; clc;

addpath(genpath('D:\Project_code\OptDSP_FSO_2\practical waterfiling-exp\OptDSP_lite-master'));

global PROG;
PROG.showMessagesLevel = 1;
initProg();

%% ===================== 用户配置 =====================

data_root = 'D:\Project_code\OptDSP_FSO_2\practical waterfiling-exp\OptDSP_lite-master\2_Data_Results';

rx_date  = '2026.06.26';
mod_name = '64QAM';
sub_name = 'sub1';
sig_idx  = 1;

rx_bin_file = fullfile(data_root, 'rx_data', rx_date, mod_name, sub_name, sprintf('%d.bin', sig_idx));

tx_root = fullfile(data_root, 'tx_3frame_6mod');
tx_frame_files = {
    fullfile(tx_root, mod_name, sub_name, sprintf('sig_%04d_frame1.mat', sig_idx))
    fullfile(tx_root, mod_name, sub_name, sprintf('sig_%04d_frame2.mat', sig_idx))
    fullfile(tx_root, mod_name, sub_name, sprintf('sig_%04d_frame3.mat', sig_idx))
};

out_root = fullfile(data_root, 'dataset_iterative_rx1_style', mod_name, sub_name);
if ~exist(out_root, 'dir'), mkdir(out_root); end

Fs_rx   = 80e9;
Fs_base = 16e9;

n_target_frames = 3;

%% ===================== OFDM 参数 =====================

SIG.nSyms = 128;

ofdm.NumberOfIFFTSamples = 256;
ofdm.NumberOfGuardTime   = 16;
ofdm.Carrier_location    = 4:126;
ofdm.Carrier_location_demo = [4:126, 132:254];
ofdm.NumberOfCarriers = length(ofdm.Carrier_location);
ofdm.NumberOfCarriers_demo = length(ofdm.Carrier_location_demo);
ofdm.size = SIG.nSyms;

zeros_head = 80;
n_fft   = ofdm.NumberOfIFFTSamples;
n_guard = ofdm.NumberOfGuardTime;
sym_len = n_fft + n_guard;

frame_len_16 = zeros_head + n_guard + 2*n_fft + sym_len*SIG.nSyms;  % 35424

%% ===================== QAM 参数 =====================

switch mod_name
    case 'QPSK',    Mq = 4;   bits = 2;
    case '16QAM',   Mq = 16;  bits = 4;
    case '32QAM',   Mq = 32;  bits = 5;
    case '64QAM',   Mq = 64;  bits = 6;
    case '128QAM',  Mq = 128; bits = 7;
    case '256QAM',  Mq = 256; bits = 8;
    otherwise, error('Unknown modulation: %s', mod_name);
end

if strcmp(mod_name, '32QAM')
    nBpS_net = bits;
else
    nBpS_net = bits - 0.2;
end

TX.SIG = setSignalParams('symRate', 8e9, 'M', Mq, ...
    'nPol', 1, 'nBpS', nBpS_net, 'nSyms', SIG.nSyms, ...
    'roll-off', 0.25, 'modulation', 'QAM');
TX.QAM = QAM_config(TX.SIG);
C = TX.QAM.IQmap;

%% ===================== 读取 TX 三帧参考 =====================

tx_refs = cell(1, 3);

for j = 1:3
    if ~exist(tx_frame_files{j}, 'file')
        error('Missing TX ref: %s', tx_frame_files{j});
    end
    tmp = load(tx_frame_files{j});
    tx_refs{j} = tmp.data_tx.';   % [123 x 128]
end

fprintf('Loaded 3 TX frame references\n');

%% ===================== 读取 RX .bin =====================

fprintf('\nReading RX bin:\n%s\n', rx_bin_file);

rx80 = read_keysight_bin_local(rx_bin_file);
rx80 = rx80(:).';

rx80 = rx80 - mean(rx80);
rx80 = rx80 ./ (rms(rx80) + eps);

fprintf('RX80 length = %d samples\n', length(rx80));

%% ===================== 重采样到 16G =====================

fprintf('Resample %.0f GS/s -> %.0f GS/s\n', Fs_rx/1e9, Fs_base/1e9);

rx16 = resample(rx80, Fs_base, Fs_rx);
rx16 = rx16 - mean(rx16);
rx16 = rx16 ./ (mean(abs(rx16)) + eps);

fprintf('RX16 length = %d samples\n', length(rx16));

wrap_len = min(length(rx16), 3*frame_len_16);
rx16_ext = [rx16, rx16(1:wrap_len)];

%% ===================== 递归式单帧提取 + 三层 SNR ======================

cursor = 1;

rx_frames = cell(1, n_target_frames);
rx_sc_all = cell(1, n_target_frames);
best_tx_id = NaN(1, n_target_frames);
frame_snr_db = NaN(1, n_target_frames);
sc_snr_all = cell(1, n_target_frames);
match_info_all = cell(1, n_target_frames);

fprintf('\nStart iterative extraction with 3-level SNR...\n');

for k = 1:n_target_frames

    fprintf('\n--- Searching RX frame %d/%d ---\n', k, n_target_frames);

    search_sig = rx16_ext(cursor:end);

    if length(search_sig) < frame_len_16
        warning('Remaining sequence too short for frame %d', k);
        break;
    end

    try
        [lts_start_rel, frame_start_rel, sync_info] = find_one_frame_start_rx1_style( ...
            search_sig, ofdm, SIG.nSyms, zeros_head);
    catch ME
        warning('Frame %d sync failed: %s', k, ME.message);
        break;
    end

    lts_start_abs   = cursor + lts_start_rel - 1;
    frame_start_abs = cursor + frame_start_rel - 1;

    fprintf('Frame %d lts_start=%d frame_start=%d edge=%d fine=%d\n', ...
        k, lts_start_abs, frame_start_abs, sync_info.edge_index, sync_info.fine_time_est);

    % 从 LTS 起点直接解调
    [rx_sc, demod_info] = demod_one_frame_from_lts_start(rx16_ext, lts_start_abs, ofdm, SIG.nSyms);

    % 取时域帧用于保存
    frame_end_abs = frame_start_abs + frame_len_16 - 1;
    if frame_end_abs <= length(rx16_ext)
        rx_frame = rx16_ext(frame_start_abs : frame_end_abs);
    else
        warning('Frame %d crosses boundary, skip', k);
        cursor = frame_start_abs + frame_len_16;
        continue;
    end
    rx_frames{k} = rx_frame;
    rx_sc_all{k} = rx_sc;

    % === 三层 SNR 计算（snr分析流程.txt）===
    [best_id, fsnr, sc_snr, match_info] = compute_snr_three_level(rx_sc, tx_refs);

    best_tx_id(k) = best_id;
    frame_snr_db(k) = fsnr;
    sc_snr_all{k} = sc_snr;
    match_info_all{k} = match_info;

    fprintf('Frame %d -> TX %d, Frame SNR=%.2f dB, nd=%d, match_mse=%.4f\n', ...
        k, best_id, fsnr, demod_info.n_use, match_info.min_mse);

    % 保存当前帧
    sample = struct();
    sample.rx_frame_16_full = single(rx_frame);
    sample.rx_sc = single(rx_sc);
    sample.subcarrier_snr_db = single(sc_snr);    % [123 x 1] 每子载波 SNR
    sample.frame_snr_db = fsnr;                   % 帧 SNR（标量）
    sample.best_tx_frame_id = best_id;
    sample.match_mse = match_info.min_mse;
    sample.match_snr_all = single(match_info.snr_all);  % 与3个TX的SNR
    sample.rx_frame_idx = k;
    sample.mod_name = mod_name;
    sample.mod_order = Mq;
    sample.sig_idx = sig_idx;
    sample.sub_name = sub_name;
    sample.rx_bin_file = rx_bin_file;
    sample.frame_start_abs_16 = frame_start_abs;
    sample.frame_len_16 = frame_len_16;
    sample.lts_start_abs = lts_start_abs;
    sample.sync_info = sync_info;
    sample.demod_info = demod_info;

    out_file = fullfile(out_root, sprintf('sig_%04d_rxframe%d.mat', sig_idx, k));
    save(out_file, 'sample', '-v7.3');

    cursor = frame_start_abs + frame_len_16;

end

%% ===================== 文件级 SNR（线性平均再转 dB）======================

valid_snr = frame_snr_db(isfinite(frame_snr_db) & frame_snr_db > 0);

if ~isempty(valid_snr)
    file_snr_db = 10*log10(mean(10.^(valid_snr/10)));
else
    file_snr_db = NaN;
end

fprintf('\n========================================\n');
fprintf('Extracted frames: %d/%d\n', sum(~cellfun(@isempty, rx_frames)), n_target_frames);
fprintf('TX match order:   ');
fprintf('%d ', best_tx_id);
fprintf('\n');
fprintf('Frame SNR (dB):   ');
fprintf('%.2f ', frame_snr_db);
fprintf('\n');
fprintf('File SNR = %.2f dB\n', file_snr_db);
fprintf('========================================\n');

%% =====================================================================
%%                       局部函数
%% =====================================================================

%% ===== 同步：返回 LTS 起点 + 帧起点 =====
function [lts_start, frame_start, info] = find_one_frame_start_rx1_style(rx, ofdm, n_syms, zeros_head)

    rx = rx(:).';

    n_fft   = ofdm.NumberOfIFFTSamples;
    n_guard = ofdm.NumberOfGuardTime;
    sym_len = n_fft + n_guard;

    symbol_bits = zeros_head + n_guard + 2*n_fft + sym_len*n_syms;

    search_len = min(length(rx), 2*symbol_bits);

    if search_len < symbol_bits
        error('input too short for one-frame sync');
    end

    search_sig = rx(1:search_len);

    [detected_packet, edge_index] = packet_edge_power_dect(search_sig, zeros_head);

    load('LongTrainSym_ini.mat', 'LongTrainSym_ini');

    LTS_f = LongTrainSym_ini(1:n_fft);
    LTS_f([1 n_fft/2+1]) = 0;

    ltrs_in = LTS_f;
    ltrs_in(1, n_fft/2+2:n_fft) = conj(ltrs_in(1, n_fft/2:-1:2));

    [fine_time_est, data_df, max_peak_long] = rx_fine_time_sync_cross_corr( ...
        detected_packet, n_guard, ltrs_in, zeros_head, 0);

    lts_start = edge_index + fine_time_est - 1;
    frame_start = lts_start - (zeros_head + n_guard);

    if frame_start < 1
        frame_start = 1;
    end

    if lts_start < 1 || lts_start + 2*n_fft - 1 > length(rx)
        error('LTS out of range: lts_start=%d', lts_start);
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

%% ===== 从已知 LTS 起点直接解调 =====
function [rx_sc, info] = demod_one_frame_from_lts_start(rx, lts_start, ofdm, n_syms)
    rx = rx(:).';

    n_fft   = ofdm.NumberOfIFFTSamples;
    n_guard = ofdm.NumberOfGuardTime;
    sym_len = n_fft + n_guard;

    carrier_loc = 4:126;

    load('LongTrainSym_ini.mat', 'LongTrainSym_ini');

    LTS_f = LongTrainSym_ini(1:n_fft);
    LTS_f([1 n_fft/2+1]) = 0;

    ltrs_in = LTS_f;
    ltrs_in(1, n_fft/2+2:n_fft) = conj(ltrs_in(1, n_fft/2:-1:2));
    LTS_f_ref = ltrs_in;

    lts1_start = lts_start;
    lts1_end   = lts_start + n_fft - 1;
    lts2_start = lts_start + n_fft;
    lts2_end   = lts_start + 2*n_fft - 1;

    if lts1_start < 1 || lts2_end > length(rx)
        error('frame too short for LTS: remaining=%d', length(rx)-lts_start+1);
    end

    lts1 = rx(lts1_start:lts1_end);
    lts2 = rx(lts2_start:lts2_end);

    pd = angle(sum(lts1(:).*conj(lts2(:))));
    cfo = pd/(2*pi*n_fft);

    n = 0:length(rx)-1;
    rx_cfo = rx .* exp(-1j*2*pi*cfo*n/n_fft);

    lts1 = rx_cfo(lts1_start:lts1_end);
    lts2 = rx_cfo(lts2_start:lts2_end);

    data_start = lts_start + 2*n_fft;
    data_end   = data_start + sym_len*n_syms - 1;

    if data_end > length(rx_cfo)
        remain = length(rx_cfo) - data_start + 1;
        nd = floor(remain / sym_len);
        if nd <= 0
            error('no complete OFDM symbols: nd=0');
        end
        n_use = min(nd, n_syms);
    else
        n_use = n_syms;
    end

    data_end = data_start + sym_len*n_use - 1;
    dp = rx_cfo(data_start:data_end);

    dm = reshape(dp, sym_len, n_use);
    dn = dm(n_guard+1:end, :);

    fd = fft(dn, n_fft, 1) ./ sqrt(n_fft);

    lts_avg = (lts1(:) + lts2(:)) / 2;
    lts_fd = fft(lts_avg, n_fft) ./ sqrt(n_fft);

    H = lts_fd ./ (LTS_f_ref(:) + 1e-12);
    H(abs(LTS_f_ref(:)) < 0.5) = 1;

    feq = fd ./ H;

    rx_sc = feq(carrier_loc, :);

    info = struct();
    info.cfo = cfo;
    info.n_use = n_use;
    info.lts_start = lts_start;
    info.data_start = data_start;
    info.data_end = data_end;
end

%% ===== 三层 SNR 计算（snr分析流程.txt）=====
% 输出：
%   best_id    — 最佳匹配的 TX 帧编号 (1/2/3)
%   frame_snr  — 帧级 SNR (dB)，盲估计
%   sc_snr     — [n_sc x 1] 每子载波 SNR (dB)
%   info       — 匹配细节（MSE、与各TX的SNR）
function [best_id, frame_snr, sc_snr, info] = compute_snr_three_level(rx_sc, tx_refs)

    n_ref = length(tx_refs);
    n_sc  = size(rx_sc, 1);

    % === 第1层：Subcarrier-level SNR（盲估计）===
    % SNR_sc(i) = 10*log10(mean(|rx_sc(i,:)|^2))
    sc_snr = 10 * log10(mean(abs(rx_sc).^2, 2));  % [n_sc x 1]

    % === 第2层：Frame-level SNR（盲估计）===
    % SNR_frame = 10*log10(mean(|rx_sc(:)|^2))
    frame_snr = 10 * log10(mean(abs(rx_sc(:)).^2));

    % === TX 匹配：计算 rx_sc 与每个 tx_ref 的 MSE ===
    % MSE 越小 → 匹配度越高
    mse_list = NaN(1, n_ref);
    snr_list = NaN(1, n_ref);  % 与各TX参考的匹配SNR

    for j = 1:n_ref
        tx_ref = tx_refs{j};
        n_sym = min(size(rx_sc, 2), size(tx_ref, 2));

        rx_use = rx_sc(:, 1:n_sym);
        tx_use = tx_ref(:, 1:n_sym);

        % 归一化后算 MSE
        rx_norm = rx_use ./ (norm(rx_use, 'fro') + eps);
        tx_norm = tx_use ./ (norm(tx_use, 'fro') + eps);

        diff = rx_norm - tx_norm;
        mse_list(j) = mean(abs(diff(:)).^2);

        % 匹配 SNR（用 rx_sc 与 tx_ref 的残差）
        err = rx_use - tx_use;
        sig_pwr = mean(abs(tx_use(:)).^2);
        noi_pwr = mean(abs(err(:)).^2);
        if noi_pwr > 0 && sig_pwr > 0
            snr_list(j) = 10 * log10(sig_pwr / noi_pwr);
        end
    end

    [~, best_id] = min(mse_list);  % MSE 最小 = 最佳匹配

    info = struct();
    info.min_mse = mse_list(best_id);
    info.mse_all = mse_list;
    info.snr_all = snr_list;
    info.best_snr = snr_list(best_id);
end

%% ===== 读 Keysight .bin（复数 I/Q）=====
function y = read_keysight_bin_local(filename)

    fid = fopen(filename, 'rb');

    if fid == -1
        error('Cannot open: %s', filename);
    end

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
    bpp = fread(fid, 1, 'int16');
    fread(fid, 1, 'int32');

    switch bpp
        case 4
            raw = fread(fid, num_points, 'float32');
        case 2
            raw = fread(fid, num_points, 'int16');
        case 1
            raw = fread(fid, num_points, 'int8');
        otherwise
            raw = fread(fid, num_points, 'double');
    end

    fclose(fid);

    raw = raw(:);
    n = floor(length(raw)/2);
    raw = raw(1:2*n);
    y = double(raw(1:2:end)) + 1j * double(raw(2:2:end));
end
