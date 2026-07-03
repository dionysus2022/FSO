% rx_direct_demod.m
% 先全信号搜索帧头 → 切分3帧 → 每帧独立调用 deOFDM 解调
% SNR: 决策指向法（无需频域参考）
clear; clear global; close all; clc;

addpath(genpath('D:\Project_code\OptDSP_FSO_2\practical waterfiling-exp\OptDSP_lite-master'));
global PROG; PROG.showMessagesLevel = 2; initProg(); RGB = fancyColors(); co=1;

%% 1. 系统参数
SIG.M = 16; SIG.symRate = 8e9/co; SIG.bitRate_net = 8e9;
SIG.modulation = 'QAM'; SIG.rollOff = 0.25; SIG.nPol = 1;
SIG.nSyms = 2^7/co; nSpS = 5; laserLW = 0e6;
FEC_rate = 1; pilotRate = 1; useCPE2 = false;

ofdm.NumberOfIFFTSamples=256; ofdm.Carrier_location=[4:126];
ofdm.Carrier_location_demo=[4:126,132:254];
ofdm.NumberOfCarriers=length(ofdm.Carrier_location);
ofdm.NumberOfCarriers_demo=length(ofdm.Carrier_location_demo);
ofdm.NumberOfGuardTime=16; Fs=10e9; Fg=10e9;
ofdm.size = SIG.nSyms;

nBpS_net = SIG.bitRate_net/(SIG.nPol*SIG.symRate*FEC_rate*pilotRate);
TX.SIG = setSignalParams('symRate',SIG.symRate,'M',SIG.M,...
    'nPol',SIG.nPol,'nBpS',nBpS_net,'nSyms',SIG.nSyms,...
    'roll-off',SIG.rollOff,'modulation',SIG.modulation);
TX.QAM = QAM_config(TX.SIG);
C = TX.QAM.IQmap; C_vec = C(:);  % 理想星座点

%% 2. 读取 .bin
data_root = 'D:\Project_code\OptDSP_FSO_2\practical waterfiling-exp\OptDSP_lite-master\2_Data_Results';
rx_bin_file = fullfile(data_root, 'rx_data', '2026.06.26', '16QAM', 'sub1', '1.bin');
scope_Fs = 80e9;
fprintf('Reading: %s\n', rx_bin_file);
fid = fopen(rx_bin_file,'rb');
if fid == -1, error('Cannot open'); end
cookie=fread(fid,2,'*char')'; version=fread(fid,2,'*char')';
file_size=fread(fid,1,'int32'); num_waveforms=fread(fid,1,'int32');
header_size=fread(fid,1,'int32'); wave_type=fread(fid,1,'int32');
num_buffers=fread(fid,1,'int32'); num_points=fread(fid,1,'int32');
count=fread(fid,1,'int32'); x_disp_range=fread(fid,1,'float32');
x_disp_orig=fread(fid,1,'float64'); x_inc=fread(fid,1,'float64');
x_orig=fread(fid,1,'float64'); x_units=fread(fid,1,'int32');
y_units=fread(fid,1,'int32'); date_str=fread(fid,16,'*char')';
time_str=fread(fid,16,'*char')'; frame_str=fread(fid,24,'*char')';
wave_str=fread(fid,16,'*char')'; time_tag=fread(fid,1,'float64');
segment_index=fread(fid,1,'uint32'); data_header_size=fread(fid,1,'int32');
buffer_type=fread(fid,1,'int16'); bytes_per_point=fread(fid,1,'int16');
buffer_size=fread(fid,1,'int32');
fprintf('  Time: %s %s, Points: %d, %d bytes/pt\n',date_str,time_str,num_points,bytes_per_point);
switch bytes_per_point
    case 4, OutputFSO=fread(fid,num_points,'float32').';
    case 2, OutputFSO=fread(fid,num_points,'int16').';
    case 1, OutputFSO=fread(fid,num_points,'int8').';
    otherwise, OutputFSO=fread(fid,num_points,'double').';
end
fclose(fid);

%% 3. 预处理
fprintf('Resample %.0f GS/s -> 16 GS/s\n', scope_Fs/1e9);
OutputFSO = resample(OutputFSO, 16e9, scope_Fs);
OutputFSO = OutputFSO - mean(OutputFSO);
AMP_rate = 1 / (sum(abs(OutputFSO)) / length(OutputFSO));
rx_signal = OutputFSO * AMP_rate;
fprintf('  Signal length: %d\n', length(rx_signal));

%% 4. 全信号搜索帧头（80个连续零）
zeros_head = 80;
n_guard = ofdm.NumberOfGuardTime;
n_fft = ofdm.NumberOfIFFTSamples;
n_syms = SIG.nSyms;
frame_len = zeros_head + n_guard + n_fft*2 + (n_fft + n_guard) * n_syms;

P = zeros(length(rx_signal) - zeros_head + 1, 1);
for n = 1:length(P)
    P(n) = sum(abs(rx_signal(n : n + zeros_head - 1)));
end
[~, edge_idx] = min(P);
fprintf('Frame header at sample: %d\n', edge_idx);

n_frames = min(floor((length(rx_signal) - edge_idx) / frame_len), 3);
fprintf('Frames: %d\n', n_frames);
if n_frames < 1, error('No complete frame.'); end

%% 5. 逐帧解调 + 决策指向SNR
n_carriers = ofdm.NumberOfCarriers;  % 123
frame_snr = {};

for f = 1:n_frames
    fprintf('\n--- Frame %d/%d ---\n', f, n_frames);
    
    % 给 deOFDM 传入足够的信号（它内部搜索需要 2*frame_len 范围）
    seg_start = edge_idx + (f-1) * frame_len;
    seg_end = min(length(rx_signal), seg_start + 2*frame_len);
    rx_seg = rx_signal(seg_start : seg_end);
    
    % deOFDM 解调
    rx_syms_flat = deOFDM(rx_seg, ofdm, SIG.nSyms);
    
    % 重塑 → (nCarriers_demo, nSyms) → 取前123载波
    rx_matrix = reshape(rx_syms_flat, SIG.nSyms, ofdm.NumberOfCarriers_demo).';
    rx_syms = rx_matrix(1:n_carriers, :).';  % (nSyms, 123)
    fprintf('  RX symbols: [%d x %d]\n', size(rx_syms,1), size(rx_syms,2));
    
    % 决策指向 SNR（无需参考）
    snr_sc = zeros(n_carriers, 1);
    for sc = 1:n_carriers
        rx_sc = rx_syms(:, sc);  % 当前载波所有符号
        % 找最近星座点
        dist2 = abs(rx_sc - C_vec.').^2;
        [min_dist2, idx] = min(dist2, [], 2);
        decisions = C_vec(idx);
        % MMSE 缩放: h = argmin |h*rx - decision|²
        h = (rx_sc' * decisions) / (rx_sc' * rx_sc + eps);
        rx_scaled = rx_sc * h;
        % 重算误差
        err = rx_scaled - decisions;
        sig_pwr = mean(abs(decisions).^2);
        noi_pwr = mean(abs(err).^2);
        if noi_pwr > 0 && sig_pwr > 0 && isfinite(noi_pwr)
            snr_sc(sc) = 10 * log10(sig_pwr / noi_pwr);
        else
            snr_sc(sc) = NaN;
        end
    end
    frame_snr{f} = snr_sc;
    valid = snr_sc(isfinite(snr_sc) & snr_sc > 0);
    if ~isempty(valid)
        fprintf('  Mean SNR = %.2f dB\n', mean(valid));
    else
        fprintf('  WARNING: All NaN\n');
    end
end

%% 6. 画图
figure('Position', [100 100 1100 600]);
colors = {'b', 'r', 'g'};
subplot(2,1,1); hold on;
for f = 1:n_frames
    snr = frame_snr{f};
    valid = isfinite(snr) & snr > 0;
    x = find(valid);
    y = snr(valid);
    if ~isempty(x)
        plot(x, y, [colors{f} '.-'], 'LineWidth', 1);
    end
end
hold off;
xlabel('Subcarrier Index'); ylabel('SNR (dB)');
title(sprintf('Per-Subcarrier SNR — 16QAM (%d frames, decision-directed)', n_frames));
legend(arrayfun(@(x) sprintf('Frame %d', x), 1:n_frames, 'UniformOutput', false));
grid on; xlim([1 n_carriers]);

subplot(2,1,2); hold on;
for f = 1:n_frames
    snr = frame_snr{f};
    valid = snr(isfinite(snr) & snr > 0);
    if ~isempty(valid)
        m = mean(valid);
        histogram(valid, 25, 'FaceColor', colors{f}, 'EdgeColor', 'none', ...
            'FaceAlpha', 0.4, 'DisplayName', sprintf('Frame %d (%.1f dB)', f, m));
    end
end
hold off;
xlabel('SNR (dB)'); ylabel('Count'); legend show; grid on;
title('SNR Distribution per Frame');

out_dir = fullfile(data_root, 'rx_data', '2026.06.26', '16QAM');
if ~exist(out_dir,'dir'), mkdir(out_dir); end
saveas(gcf, fullfile(out_dir, 'snr_3frames.png'));
fprintf('\nSaved: %s\n', fullfile(out_dir, 'snr_3frames.png'));
fprintf('Done.\n');
