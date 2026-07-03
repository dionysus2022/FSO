% rx3_independent_frame_ref.m
% 功能：接收 AWG 循环发送的 3 帧拼接信号；发送端每帧 data_tx 单独保存；
%       RX 自动处理跨 AWG 周期的 3 帧，例如 3->1->2，并输出子载波 SNR 曲线。
%
% 发送端建议保存：
%   sig_0001_frame1.mat  变量 data_tx
%   sig_0001_frame2.mat  变量 data_tx
%   sig_0001_frame3.mat  变量 data_tx
% 仍建议保存拼接波形：
%   sig_0001.txt         变量 InputFSO_all，用于估计真实 80G 帧长和同步

clear; clear global; close all; clc;

addpath(genpath('D:\Project_code\OptDSP_FSO_2\practical waterfiling-exp\OptDSP_lite-master'));

global PROG;
PROG.showMessagesLevel = 2;
initProg();
RGB = fancyColors(); %#ok<NASGU>
co = 1;

%% ===================== 0. 用户参数 =====================
data_root = 'D:\Project_code\OptDSP_FSO_2\practical waterfiling-exp\OptDSP_lite-master\2_Data_Results';

mod_name = '16QAM';      % QPSK / 16QAM / 32QAM / 64QAM / 128QAM / 256QAM
SIG.M    = 16;           % 4/16/32/64/128/256
sub_name = 'sub1';
rx_date  = '2026.06.26';

rx_bin_idx = 1;          % rx_data/.../1.bin
sig_idx    = 1;          % tx_3frame_6mod/.../sig_0001_frame1.mat 等
n_frames_expected = 3;

scope_Fs = 80e9;         % 示波器采样率
ofdm_Fs  = 16e9;         % deOFDM 输入采样率

% 同步方式：推荐 true。需要 sig_0001.txt，即 tx3 拼接后的 InputFSO_all。
use_tx_txt_sync = true;
rx_start_80_manual = 1;  % 不使用同步时手动指定第一段起点

% 为了让 deOFDM 内部同步有余量，每段默认取 2 帧长度。
% 但当前帧自身必须完整；如果靠近文件尾部，会自动回退到 AWG 前一周期。
rx_seg_len_frames_for_deofdm = 3;

%% ===================== 1. 系统参数，与 TX 保持一致 =====================
SIG.symRate = 8e9/co;
SIG.bitRate_net = 8e9;
SIG.modulation = 'QAM';
SIG.rollOff = 0.25;
SIG.nPol = 1;
SIG.nSyms = 2^7/co;
nSpS = 5; %#ok<NASGU>
laserLW = 0e6; %#ok<NASGU>
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

bits = log2(SIG.M);
if bits <= 2
    nBpS_net = bits;
else
    nBpS_net = bits - 0.2;   % 必须与 tx3 保持一致
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

%% ===================== 2. 路径 =====================
rx_bin_file = fullfile(data_root, 'rx_data', rx_date, mod_name, sub_name, sprintf('%d.bin', rx_bin_idx));
tx_dir = fullfile(data_root, 'tx_3frame_6mod', mod_name, sub_name);
ref_txt_file = fullfile(tx_dir, sprintf('sig_%04d.txt', sig_idx));
out_dir = fullfile(data_root, 'rx_data', rx_date, mod_name, sub_name);
if ~exist(out_dir, 'dir'), mkdir(out_dir); end

fprintf('\nRX bin : %s\n', rx_bin_file);
fprintf('TX dir : %s\n', tx_dir);
fprintf('TX txt : %s\n', ref_txt_file);

%% ===================== 3. 读取 RX .bin，保留 80G =====================
OutputFSO80 = read_keysight_bin_local(rx_bin_file);
OutputFSO80 = OutputFSO80(:);
OutputFSO80 = OutputFSO80 - mean(OutputFSO80);
fprintf('\nRead RX samples: %d at %.0f GS/s\n', length(OutputFSO80), scope_Fs/1e9);

%% ===================== 4. 读取 3 个独立 TX 参考 mat =====================
tx_frame_ref = cell(1, n_frames_expected);
for k = 1:n_frames_expected
    fmat = fullfile(tx_dir, sprintf('sig_%04d_frame%d.mat', sig_idx, k));
    if ~exist(fmat, 'file')
        error(['找不到独立帧参考文件：%s\n' ...
               '请先在 tx3 中保存 sig_XXXX_frame1.mat / frame2.mat / frame3.mat。'], fmat);
    end
    tmp = load(fmat);
    if isfield(tmp, 'data_tx')
        data_tx = tmp.data_tx;
    elseif isfield(tmp, 'tx_frame')
        data_tx = tmp.tx_frame;
    else
        error('文件 %s 中没有 data_tx 或 tx_frame 变量。', fmat);
    end
    if size(data_tx, 1) ~= SIG.nSyms
        error('文件 %s 的 data_tx 行数不是 SIG.nSyms=%d，请检查保存方向。', fmat, SIG.nSyms);
    end
    tx_frame_ref{k} = data_tx.';  % [123 x 128]
    fprintf('Loaded TX frame %d ref: %s, size [%d x %d]\n', k, fmat, size(tx_frame_ref{k},1), size(tx_frame_ref{k},2));
end

%% ===================== 5. 读取 TX 拼接 txt，得到真实帧长 =====================
if exist(ref_txt_file, 'file')
    tx_ref80_all = load_ascii_complex_local(ref_txt_file);
    tx_ref80_all = tx_ref80_all(:);
    frame_len_80 = floor(length(tx_ref80_all) / n_frames_expected);
    fprintf('Frame length from TX txt: %d samples at 80G\n', frame_len_80);
else
    frame_len_16_nominal = ofdm.NumberOfGuardTime + 2*ofdm.NumberOfIFFTSamples + ...
        (ofdm.NumberOfIFFTSamples + ofdm.NumberOfGuardTime) * SIG.nSyms;
    frame_len_80_nominal = round(frame_len_16_nominal * scope_Fs / ofdm_Fs);
    frame_len_80 = ceil(frame_len_80_nominal / 256) * 256;
    tx_ref80_all = [];
    fprintf('WARNING: 没有 TX txt，使用理论帧长估计：%d samples at 80G\n', frame_len_80);
end

if frame_len_80 <= 0
    error('frame_len_80 计算错误。');
end

%% ===================== 6. 同步并选择可完整覆盖 3 帧的起点 =====================
if use_tx_txt_sync && exist(ref_txt_file, 'file')
    % 用 3 个 TX 帧模板分别相关，选一个相关峰最高的帧作为锚点。
    best_metric = -Inf;
    best_detect_pos = NaN;
    best_detect_frame_id = NaN;

    for k = 1:n_frames_expected
        tx_k80 = tx_ref80_all((k-1)*frame_len_80 + (1:frame_len_80));
        [pos_k, metric_k] = find_start_by_corr_local(OutputFSO80, tx_k80);
        fprintf('Sync candidate: TX frame %d detected at RX sample %d, metric %.4f\n', k, pos_k, metric_k);
        if metric_k > best_metric
            best_metric = metric_k;
            best_detect_pos = pos_k;
            best_detect_frame_id = k;
        end
    end

    fprintf('Best anchor: TX frame %d at RX sample %d\n', best_detect_frame_id, best_detect_pos);

    % 如果检测到的是第 k 帧，则该 AWG 周期中 frame1 的理论起点为：
    % block_start = detect_pos - (k-1)*frame_len_80。
    block_start_80 = best_detect_pos - (best_detect_frame_id - 1) * frame_len_80;

    % 由于 AWG 循环播放，如果 block_start 太靠后导致放不下 3 帧，就整体往前回退一个或多个帧长。
    % 回退一帧后，TX 参考顺序也要循环变化，例如 frame1 起点回退 1 帧 => 顺序 3,1,2。
    while block_start_80 + n_frames_expected*frame_len_80 - 1 > length(OutputFSO80)
        block_start_80 = block_start_80 - frame_len_80;
    end

    % 如果回退过头导致起点小于 1，则向后移动一个帧长，直到合法。
    while block_start_80 < 1
        block_start_80 = block_start_80 + frame_len_80;
    end

    % 计算当前 block_start 对应的 TX 帧编号。
    % block_start = 周期 frame1 起点 + offset*frame_len，offset 可为负。
    offset_frames = round((block_start_80 - (best_detect_pos - (best_detect_frame_id - 1)*frame_len_80)) / frame_len_80);
    first_tx_frame_id = mod(offset_frames, n_frames_expected) + 1;
    if first_tx_frame_id <= 0, first_tx_frame_id = first_tx_frame_id + n_frames_expected; end

else
    block_start_80 = rx_start_80_manual;
    first_tx_frame_id = 1;
end

if block_start_80 < 1 || block_start_80 + n_frames_expected*frame_len_80 - 1 > length(OutputFSO80)
    error(['无法在当前 RX 文件中找到完整 3 帧。\n' ...
           'block_start_80=%d, frame_len_80=%d, RX length=%d。\n' ...
           '请增加示波器记录长度或调整触发位置。'], block_start_80, frame_len_80, length(OutputFSO80));
end

tx_order = mod((first_tx_frame_id-1) + (0:n_frames_expected-1), n_frames_expected) + 1;

fprintf('\nSelected RX block start: %d\n', block_start_80);
fprintf('TX reference order for 3 RX segments: %s\n', mat2str(tx_order));
fprintf('Frames to process: %d\n', n_frames_expected);

%% ===================== 7. 逐帧 deOFDM + SNR =====================
n_carriers = ofdm.NumberOfCarriers;
SNR_frame = NaN(n_frames_expected, n_carriers);
BER_frame = NaN(n_frames_expected, n_carriers);
EVM_frame = NaN(n_frames_expected, n_carriers);
rx_segment_start_80 = NaN(n_frames_expected, 1);
rx_segment_end_80 = NaN(n_frames_expected, 1);

for f = 1:n_frames_expected
    tx_id = tx_order(f);
    fprintf('\n--- RX segment %d/%d, use TX frame %d reference ---\n', f, n_frames_expected, tx_id);

    seg_start_80 = block_start_80 + (f-1)*frame_len_80;
    seg_end_80 = min(length(OutputFSO80), seg_start_80 + rx_seg_len_frames_for_deofdm*frame_len_80 - 1);

    % 当前帧自身必须完整，否则不进入 deOFDM。
    if seg_start_80 + frame_len_80 - 1 > length(OutputFSO80)
        error('第 %d 段当前帧不完整，不能 deOFDM。', f);
    end

    rx_segment_start_80(f) = seg_start_80;
    rx_segment_end_80(f) = seg_end_80;

    rx_seg80 = OutputFSO80(seg_start_80:seg_end_80).';
    rx_seg16 = resample(rx_seg80, ofdm_Fs, scope_Fs);
    rx_seg16 = rx_seg16 - mean(rx_seg16);
    rx_seg16 = rx_seg16 ./ (mean(abs(rx_seg16)) + eps);

    fprintf('  Segment: 80G [%d, %d], len80=%d, len16=%d\n', ...
        seg_start_80, seg_end_80, length(rx_seg80), length(rx_seg16));

    rx_flat = deOFDM(rx_seg16, ofdm, SIG.nSyms);

    if useCPE2
        [rx_flat, DSP.CPE2] = carrierPhaseEstimation(rx_flat, tx_frame_ref{tx_id}, DSP.CPE2, C);
    end

    rx_demo = reshape(rx_flat, SIG.nSyms, ofdm.NumberOfCarriers_demo).';
    rx_sc = rx_demo(1:n_carriers, :);      % [123 x 128]
    tx_frame = tx_frame_ref{tx_id};        % [123 x 128]

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

%% ===================== 8. 画图保存 =====================
valid_snr = SNR_frame;
valid_snr(~isfinite(valid_snr) | valid_snr <= 0) = NaN;
SNR_mean = mean(valid_snr, 1, 'omitnan');

figure('Position', [100 100 1200 650]);
hold on;
for f = 1:n_frames_expected
    plot(1:n_carriers, valid_snr(f, :), '.-', 'LineWidth', 1, ...
        'DisplayName', sprintf('RX seg %d / TX frame %d', f, tx_order(f)));
end
plot(1:n_carriers, SNR_mean, 'k-', 'LineWidth', 2.0, 'DisplayName', 'Mean of 3 segments');
hold off;
grid on;
xlim([1 n_carriers]);
xlabel('Subcarrier Index');
ylabel('SNR (dB)');
title(sprintf('%s Per-Subcarrier SNR, AWG cyclic 3-frame RX', mod_name));
legend('Location', 'best');

fig_file = fullfile(out_dir, sprintf('snr_3frames_independent_%s_sig%04d.png', mod_name, sig_idx));
mat_file = fullfile(out_dir, sprintf('snr_3frames_independent_%s_sig%04d.mat', mod_name, sig_idx));
saveas(gcf, fig_file);
save(mat_file, 'SNR_frame', 'SNR_mean', 'BER_frame', 'EVM_frame', ...
    'block_start_80', 'frame_len_80', 'tx_order', 'rx_segment_start_80', 'rx_segment_end_80', ...
    'mod_name', 'sig_idx', 'rx_bin_file', 'tx_dir');

fprintf('\nSaved figure: %s\n', fig_file);
fprintf('Saved data  : %s\n', mat_file);
fprintf('Done.\n');

%% ===================== 局部函数 =====================
function y = read_keysight_bin_local(filename)
    fid = fopen(filename, 'rb');
    if fid == -1, error('Cannot open file: %s', filename); end

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
    date_str = fread(fid, 16, '*char')';
    time_str = fread(fid, 16, '*char')';
    fread(fid, 24, '*char')';
    fread(fid, 16, '*char')';
    fread(fid, 1, 'float64');
    fread(fid, 1, 'uint32');
    fread(fid, 1, 'int32');
    fread(fid, 1, 'int16');
    bytes_per_point = fread(fid, 1, 'int16');
    fread(fid, 1, 'int32');

    fprintf('  Time: %s %s, Points: %d, %d bytes/pt\n', date_str, time_str, num_points, bytes_per_point);

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

function [start_idx, best_metric] = find_start_by_corr_local(rx, tx_first)
    rx = rx(:);
    tx_first = tx_first(:);

    decim = 20;
    rx_d = abs(rx(1:decim:end));
    tx_d = abs(tx_first(1:decim:end));

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

    win = 5 * decim;
    s1 = max(1, start_coarse - win);
    s2 = min(length(rx) - length(tx_first) + 1, start_coarse + win);
    if s2 <= s1
        start_idx = start_coarse;
        best_metric = max(abs(c)) / length(tx_d);
        return;
    end

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

    [best_metric, imax] = max(metric);
    start_idx = s1 + imax - 1;
end
