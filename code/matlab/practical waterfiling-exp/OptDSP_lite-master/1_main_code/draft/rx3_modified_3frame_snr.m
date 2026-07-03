% rx3_modified_3frame_snr.m
% 功能：接收 tx3 生成的 3 帧拼接信号，按帧切分 -> 每帧 deOFDM -> 解调 -> 输出每个子载波 SNR 曲线
% 关键修正：
% 1) tx3 没有插入 80 个零帧头，不能再用 zeros_head 搜索帧头；
% 2) tx3 是每帧在 80 GS/s 端做 256 点对齐后再拼接，所以必须优先在 80 GS/s 原始采样率下切分；
% 3) RX 的 QAM_config 参数必须和 tx3 一致，高阶 QAM 使用 bits - 0.2 的 nBpS_net；
% 4) SNR 默认使用 rx1 的参考 EVM_eval 方法，并按 3 帧分别计算子载波 SNR。

clear; clear global; close all; clc;

addpath(genpath('D:\Project_code\OptDSP_FSO_2\practical waterfiling-exp\OptDSP_lite-master'));

global PROG;
PROG.showMessagesLevel = 2;
initProg();
RGB = fancyColors();
co = 1;

%% ===================== 0. 用户需要改的参数 =====================
data_root = 'D:\Project_code\OptDSP_FSO_2\practical waterfiling-exp\OptDSP_lite-master\2_Data_Results';

mod_name = '16QAM';      % 可改：QPSK / 16QAM / 32QAM / 64QAM / 128QAM / 256QAM
SIG.M    = 16;           % 和 mod_name 对应：4/16/32/64/128/256
sub_name = 'sub1';
rx_bin_idx = 1;          % 接收端 .bin 文件编号，例如 1.bin
sig_idx    = 1;          % 发送端 sig_0001.mat / sig_0001.txt 编号；sub2 通常从 26 开始，sub3 从 51，sub4 从 76

n_frames_expected = 3;
scope_Fs = 80e9;         % 示波器采样率
ofdm_Fs  = 16e9;         % deOFDM 前的目标采样率，必须和 tx3 中 OFDM 基带采样率一致

% 是否使用发送端 txt 做相关同步。推荐 true。
% 如果没有保存 sig_xxxx.txt，程序会退化为从 rx_start_80_manual 开始按长度切分。
use_tx_txt_sync = true;
rx_start_80_manual = 1;  % 不使用 txt 同步时，从原始 80G 波形的这个点开始切第一帧

%% ===================== 1. 系统参数，必须与 tx3 一致 =====================
SIG.symRate = 8e9/co;
SIG.bitRate_net = 8e9;
SIG.modulation = 'QAM';
SIG.rollOff = 0.25;
SIG.nPol = 1;
SIG.nSyms = 2^7/co;
nSpS = 5;
laserLW = 0e6;
FEC_rate = 1;
pilotRate = 1;
useCPE2 = false;

ofdm.NumberOfIFFTSamples = 256;
ofdm.Carrier_location = 4:126;
ofdm.Carrier_location_demo = [4:126, 132:254];
ofdm.NumberOfCarriers = length(ofdm.Carrier_location);
ofdm.NumberOfCarriers_demo = length(ofdm.Carrier_location_demo);
ofdm.NumberOfGuardTime = 16;
ofdm.size = SIG.nSyms;

% tx3 中高阶调制使用 bits - 0.2，RX 端必须一致
bits = log2(SIG.M);
if bits <= 2
    nBpS_net = bits;
else
    nBpS_net = bits - 0.2;
end

TX.SIG = setSignalParams('symRate', SIG.symRate, 'M', SIG.M, ...
    'nPol', SIG.nPol, 'nBpS', nBpS_net, 'nSyms', SIG.nSyms, ...
    'roll-off', SIG.rollOff, 'modulation', SIG.modulation);
TX.QAM = QAM_config(TX.SIG);

TX.PILOTS.active = true;
TX.PILOTS.rate = pilotRate;
TX.PILOTS.option = 'outerQPSK';
TX.FEC.active = false;
TX.FEC.rate = FEC_rate;
TX.FEC.nIter = 50;
TX.PCS.method = 'CCDM';

C = TX.QAM.IQmap;

DSP.MF.type = 'RRC';
DSP.MF.rollOff = TX.SIG.rollOff;
DSP.CPE1.method = 'pilot-based:optimized';
DSP.CPE1.decision = 'data-aided';
DSP.CPE1.nTaps_min = 1;
DSP.CPE1.nTaps_max = 201;
DSP.CPE1.PILOTS = TX.PILOTS;
DSP.CPE2.method = 'BPS';
DSP.CPE2.nTaps = 22;
DSP.CPE2.nTaps_min = 1;
DSP.CPE2.nTaps_max = 501;
DSP.CPE2.nTestPhases = 10;
DSP.CPE2.angleInterval = pi/8;
DSP.DEMAPPER.normMethod = 'MMSE';

%% ===================== 2. 文件路径 =====================
rx_bin_file = fullfile(data_root, 'rx_data', '2026.06.26', mod_name, sub_name, sprintf('%d.bin', rx_bin_idx));
ref_mat_file = fullfile(data_root, 'tx_3frame_6mod', mod_name, sub_name, sprintf('sig_%04d.mat', sig_idx));
ref_txt_file = fullfile(data_root, 'tx_3frame_6mod', mod_name, sub_name, sprintf('sig_%04d.txt', sig_idx));

out_dir = fullfile(data_root, 'rx_data', '2026.06.26', mod_name, sub_name);
if ~exist(out_dir, 'dir'), mkdir(out_dir); end

fprintf('\nRX bin : %s\n', rx_bin_file);
fprintf('TX mat : %s\n', ref_mat_file);
fprintf('TX txt : %s\n', ref_txt_file);

%% ===================== 3. 读取接收 .bin，保持 80G，不要先 resample =====================
OutputFSO80 = read_keysight_bin_local(rx_bin_file);
OutputFSO80 = OutputFSO80(:);       % 列向量
OutputFSO80 = OutputFSO80 - mean(OutputFSO80);

fprintf('\nRead RX samples: %d at %.0f GS/s\n', length(OutputFSO80), scope_Fs/1e9);

%% ===================== 4. 读取发送端参考符号 data_tx_all =====================
ref = load(ref_mat_file);
if isfield(ref, 'data_tx_all')
    data_tx_all = ref.data_tx_all;
elseif isfield(ref, 'data_tx')
    % 兼容单帧旧文件
    data_tx_all = ref.data_tx;
else
    error('参考 mat 文件中没有 data_tx_all 或 data_tx。请检查 tx3 保存变量名。');
end

if size(data_tx_all, 1) < n_frames_expected * SIG.nSyms
    error('data_tx_all 行数不足：当前 %d 行，至少需要 %d 行。', size(data_tx_all, 1), n_frames_expected * SIG.nSyms);
end

%% ===================== 5. 确定每帧在 80G 下的长度 =====================
% 优先从 tx3 保存的 txt 得到真实长度，因为 tx3 每帧做了 256 点补零对齐。
% 如果没有 txt，则按 OFDM 理论长度估计。
if exist(ref_txt_file, 'file')
    tx_ref80 = load_ascii_complex_local(ref_txt_file);
    tx_ref80 = tx_ref80(:);
    frame_len_80 = floor(length(tx_ref80) / n_frames_expected);
    fprintf('Frame length from TX txt: %d samples at 80G\n', frame_len_80);
else
    frame_len_16_nominal = ofdm.NumberOfGuardTime + 2*ofdm.NumberOfIFFTSamples + ...
        (ofdm.NumberOfIFFTSamples + ofdm.NumberOfGuardTime) * SIG.nSyms;
    frame_len_80_nominal = round(frame_len_16_nominal * scope_Fs / ofdm_Fs);
    frame_len_80 = ceil(frame_len_80_nominal / 256) * 256;
    tx_ref80 = [];
    fprintf('WARNING: TX txt 不存在，使用理论帧长估计：%d samples at 80G\n', frame_len_80);
end

if frame_len_80 <= 0
    error('frame_len_80 计算错误。');
end

%% ===================== 6. 找到第一帧起点，然后在 80G 下切分 =====================
if use_tx_txt_sync && exist(ref_txt_file, 'file')
    % 用第一帧发送波形做低复杂度相关同步。
    tx_first80 = tx_ref80(1:frame_len_80);
    rx_start_80 = find_start_by_corr_local(OutputFSO80, tx_first80);
    fprintf('Frame-1 start by TX waveform correlation: %d\n', rx_start_80);
else
    rx_start_80 = rx_start_80_manual;
    fprintf('Frame-1 start by manual setting: %d\n', rx_start_80);
end

if rx_start_80 < 1 || rx_start_80 + frame_len_80 - 1 > length(OutputFSO80)
    error('第一帧切分范围越界。请检查 rx_start_80 或相关同步结果。');
end

n_frames = n_frames_expected;  % 固定3帧，切分越界由 seg_end 处理
if n_frames < n_frames_expected
    warning('接收数据中只能切出 %d 帧，不足 %d 帧。', n_frames, n_frames_expected);
end

fprintf('Frames to process: %d\n', n_frames);

%% ===================== 7. 逐帧 resample -> deOFDM -> 解调 -> SNR =====================
n_carriers = ofdm.NumberOfCarriers;
SNR_frame = NaN(n_frames, n_carriers);
BER_frame = NaN(n_frames, n_carriers);
EVM_frame = NaN(n_frames, n_carriers);

for f = 1:n_frames
    fprintf('\n--- Frame %d/%d ---\n', f, n_frames);

    seg_start_80 = rx_start_80 + (f-1) * frame_len_80;
    if f == n_frames
        % 最后一帧取到信号末尾即可，16G补零会补齐不足
        seg_end_80 = length(OutputFSO80);
    else
        seg_end_80 = min(length(OutputFSO80), seg_start_80 + 2 * frame_len_80 - 1);
    end
    rx_seg80 = OutputFSO80(seg_start_80:seg_end_80).';   % 行向量

    % 每帧单独 resample，避免 80G 端补零长度不是 5 的整数倍导致的帧边界小数漂移
    rx_seg16 = resample(rx_seg80, ofdm_Fs, scope_Fs);
    rx_seg16 = rx_seg16 - mean(rx_seg16);
    rx_seg16 = rx_seg16 ./ (mean(abs(rx_seg16)) + eps);

    fprintf('  Segment: 80G [%d, %d], len80=%d, len16=%d\n', ...
        seg_start_80, seg_end_80, length(rx_seg80), length(rx_seg16));

    % 补零到 2*symbol_bits，确保 deOFDM 帧头搜索不越界
    min_len = 2 * (80 + 16 + 256*2 + (256+16)*128);
    if length(rx_seg16) < min_len
        rx_seg16 = [rx_seg16, zeros(1, min_len - length(rx_seg16))];
    end

    % deOFDM
    rx_flat = deOFDM(rx_seg16, ofdm, SIG.nSyms);

    if pilotRate < 1
        % 当前 pilotRate=1，通常不会进入；保留兼容结构
        tx_ref_tmp = data_tx_all((f-1)*SIG.nSyms + (1:SIG.nSyms), :).';
        [rx_flat, DSP.CPE1] = carrierPhaseEstimation(rx_flat, tx_ref_tmp, DSP.CPE1);
    end

    if useCPE2
        tx_ref_tmp = data_tx_all((f-1)*SIG.nSyms + (1:SIG.nSyms), :).';
        [rx_flat, DSP.CPE2] = carrierPhaseEstimation(rx_flat, tx_ref_tmp, DSP.CPE2, C);
    end

    % reshape：deOFDM 输出按 rx1 的方式恢复为 [carrier_demo x nSyms]
    rx_demo = reshape(rx_flat, SIG.nSyms, ofdm.NumberOfCarriers_demo).';
    rx_sc = rx_demo(1:n_carriers, :);       % [123 x 128]

    tx_frame = data_tx_all((f-1)*SIG.nSyms + (1:SIG.nSyms), :).';  % [123 x 128]

    txafdem_matrix = NaN(n_carriers, SIG.nSyms);
    BER_sc = NaN(n_carriers, 1);

    for sc = 1:n_carriers
        DSP.DEMAPPER.N0 = 0;
        [DSP.DEMAPPER, txafdem] = symDemapper(rx_sc(sc, :), tx_frame(sc, :), C, DSP.DEMAPPER);
        [BER, ~] = BER_eval(DSP.DEMAPPER.txBits, DSP.DEMAPPER.rxBits);
        BER_sc(sc) = BER;
        txafdem_matrix(sc, :) = txafdem;
    end

    [EVM_sc, SNR_sc] = EVM_eval(rx_sc, txafdem_matrix);

    SNR_frame(f, :) = SNR_sc(:).';
    BER_frame(f, :) = BER_sc(:).';
    EVM_frame(f, :) = EVM_sc(:).';

    valid = SNR_sc(isfinite(SNR_sc) & SNR_sc > 0);
    if ~isempty(valid)
        avg_snr_linear = 10*log10(mean(10.^(valid/10)));
        fprintf('  Mean SNR arithmetic = %.2f dB\n', mean(valid));
        fprintf('  Mean SNR linear     = %.2f dB\n', avg_snr_linear);
        fprintf('  Mean BER            = %.3e\n', mean(BER_sc, 'omitnan'));
    else
        fprintf('  WARNING: 当前帧没有有效 SNR。\n');
    end
end

%% ===================== 8. 画图并保存 =====================
valid_snr = SNR_frame;
valid_snr(~isfinite(valid_snr) | valid_snr <= 0) = NaN;
SNR_mean = mean(valid_snr, 1, 'omitnan');

figure('Position', [100 100 1200 650]);
hold on;
for f = 1:n_frames
    plot(1:n_carriers, valid_snr(f, :), '.-', 'LineWidth', 1, ...
        'DisplayName', sprintf('Frame %d', f));
end
plot(1:n_carriers, SNR_mean, 'k-', 'LineWidth', 2.0, 'DisplayName', 'Mean of 3 frames');
hold off;
grid on;
xlim([1 n_carriers]);
xlabel('Subcarrier Index');
ylabel('SNR (dB)');
title(sprintf('%s Per-Subcarrier SNR, %d frames', mod_name, n_frames));
legend('Location', 'best');

fig_file = fullfile(out_dir, sprintf('snr_3frames_%s_sig%04d.png', mod_name, sig_idx));
mat_file = fullfile(out_dir, sprintf('snr_3frames_%s_sig%04d.mat', mod_name, sig_idx));
saveas(gcf, fig_file);
save(mat_file, 'SNR_frame', 'SNR_mean', 'BER_frame', 'EVM_frame', ...
    'rx_start_80', 'frame_len_80', 'mod_name', 'sig_idx', 'rx_bin_file', 'ref_mat_file');

fprintf('\nSaved figure: %s\n', fig_file);
fprintf('Saved data  : %s\n', mat_file);
fprintf('Done.\n');

%% ===================== 本脚本用到的局部函数 =====================
function y = read_keysight_bin_local(filename)
    fid = fopen(filename, 'rb');
    if fid == -1
        error('Cannot open file: %s', filename);
    end

    cookie = fread(fid, 2, '*char')'; %#ok<NASGU>
    version = fread(fid, 2, '*char')'; %#ok<NASGU>
    file_size = fread(fid, 1, 'int32'); %#ok<NASGU>
    num_waveforms = fread(fid, 1, 'int32'); %#ok<NASGU>
    header_size = fread(fid, 1, 'int32'); %#ok<NASGU>
    wave_type = fread(fid, 1, 'int32'); %#ok<NASGU>
    num_buffers = fread(fid, 1, 'int32'); %#ok<NASGU>
    num_points = fread(fid, 1, 'int32');
    count = fread(fid, 1, 'int32'); %#ok<NASGU>
    x_disp_range = fread(fid, 1, 'float32'); %#ok<NASGU>
    x_disp_orig = fread(fid, 1, 'float64'); %#ok<NASGU>
    x_inc = fread(fid, 1, 'float64'); %#ok<NASGU>
    x_orig = fread(fid, 1, 'float64'); %#ok<NASGU>
    x_units = fread(fid, 1, 'int32'); %#ok<NASGU>
    y_units = fread(fid, 1, 'int32'); %#ok<NASGU>
    date_str = fread(fid, 16, '*char')';
    time_str = fread(fid, 16, '*char')';
    frame_str = fread(fid, 24, '*char')'; %#ok<NASGU>
    wave_str = fread(fid, 16, '*char')'; %#ok<NASGU>
    time_tag = fread(fid, 1, 'float64'); %#ok<NASGU>
    segment_index = fread(fid, 1, 'uint32'); %#ok<NASGU>
    data_header_size = fread(fid, 1, 'int32'); %#ok<NASGU>
    buffer_type = fread(fid, 1, 'int16'); %#ok<NASGU>
    bytes_per_point = fread(fid, 1, 'int16');
    buffer_size = fread(fid, 1, 'int32'); %#ok<NASGU>

    fprintf('  Time: %s %s, Points: %d, %d bytes/pt\n', ...
        date_str, time_str, num_points, bytes_per_point);

    switch bytes_per_point
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
        x = complex(tmp(:, 1), tmp(:, 2));
    else
        x = tmp(:);
    end
end

function start_idx = find_start_by_corr_local(rx, tx_first)
    rx = rx(:);
    tx_first = tx_first(:);

    % 降采样做粗同步，避免长序列 xcorr 太慢
    decim = 20;
    rx_d = abs(rx(1:decim:end));
    tx_d = abs(tx_first(1:decim:end));

    % 为避免模板太长导致相关过慢，只取第一帧前半段做同步
    max_tx_len = min(length(tx_d), 12000);
    tx_d = tx_d(1:max_tx_len);

    rx_d = rx_d - mean(rx_d);
    tx_d = tx_d - mean(tx_d);
    rx_d = rx_d ./ (std(rx_d) + eps);
    tx_d = tx_d ./ (std(tx_d) + eps);

    if length(rx_d) < length(tx_d)
        error('接收序列短于同步模板，无法相关同步。');
    end

    c = conv(rx_d, flipud(tx_d), 'valid');
    [~, idx_d] = max(abs(c));
    start_coarse = (idx_d - 1) * decim + 1;

    % 在粗同步附近做一次细搜索
    win = 5 * decim;
    s1 = max(1, start_coarse - win);
    s2 = min(length(rx) - length(tx_first) + 1, start_coarse + win);

    if s2 <= s1
        start_idx = start_coarse;
        return;
    end

    % 细同步也用包络，减少幅度极性影响
    tx_env = abs(tx_first);
    tx_env = tx_env - mean(tx_env);
    tx_env = tx_env ./ (norm(tx_env) + eps);

    metric = zeros(s2-s1+1, 1);
    for k = s1:s2
        r = abs(rx(k:k+length(tx_first)-1));
        r = r - mean(r);
        r = r ./ (norm(r) + eps);
        metric(k-s1+1) = abs(r' * tx_env);
    end

    [~, imax] = max(metric);
    start_idx = s1 + imax - 1;
end
