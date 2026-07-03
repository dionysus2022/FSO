%% =========================================================
%  signal_pipeline_v1.m
%  功能：80G原始信号 → 同步 → 3帧切分 → 解调 → SNR分析
%% =========================================================

clear; clc; close all;

addpath(genpath('D:\Project_code\OptDSP_FSO_2\practical waterfiling-exp\OptDSP_lite-master'));

%% ================== 参数区 ==================
rx_bin = 'YOUR_RX_FILE.bin';

Fs_rx   = 80e9;
Fs_base = 16e9;

n_frames = 3;

ofdm.Nfft = 256;
ofdm.Ncp  = 16;

zeros_head = 80;
n_fft = ofdm.Nfft;
n_guard = ofdm.Ncp;

sym_len = n_fft + n_guard;
frame_sym = 128;

frame_len_80 = zeros_head + n_guard + 2*n_fft + sym_len * frame_sym;

%% ================== 1. 读取信号 ==================
rx80 = read_keysight_bin(rx_bin);
rx80 = rx80(:).';

rx80 = rx80 - mean(rx80);
rx80 = rx80 ./ (rms(rx80) + eps);

fprintf("RX loaded: %d samples\n", length(rx80));

%% ================== 2. 粗同步（能量法） ==================
win_len = frame_len_80;
energy = movmean(abs(rx80).^2, win_len);

[~, start_idx] = min(energy);

fprintf("Coarse sync index: %d\n", start_idx);

%% ================== 3. 帧对齐（关键） ==================
frame_start = start_idx;

frames_80 = cell(1, n_frames);

for k = 1:n_frames

    s = frame_start + (k-1)*frame_len_80;
    e = s + frame_len_80 - 1;

    if e > length(rx80)
        warning("Frame %d incomplete", k);
        break;
    end

    frames_80{k} = rx80(s:e);

end

fprintf("Frames extracted: %d\n", sum(~cellfun(@isempty, frames_80)));

%% ================== 4. 每帧处理 ==================
SNR_frame = zeros(1, n_frames);

for k = 1:n_frames

    if isempty(frames_80{k})
        continue;
    end

    rx_f = frames_80{k};

    %% ---- 4.1 resample ----
    rx16 = resample(rx_f, Fs_base, Fs_rx);
    rx16 = rx16 - mean(rx16);
    rx16 = rx16 ./ (rms(rx16) + eps);

    %% ---- 4.2 LTS同步 ----
    load('LongTrainSym_ini.mat','LongTrainSym_ini');

    LTS = LongTrainSym_ini(1:n_fft);
    LTS([1 n_fft/2+1]) = 0;
    LTS_t = ifft(LTS);

    xc = abs(conv(rx16, flipud(conj(LTS_t)),'valid'));
    [~, pk] = max(xc);

    frm_start = pk;

    if frm_start + frame_len_16 > length(rx16)
        warning("Frame %d sync fail", k);
        continue;
    end

    rx_frame = rx16(frm_start:frm_start+frame_len_16-1);

    %% ---- 4.3 FFT + 子载波 ----
    data_start = zeros_head + n_guard + 2*n_fft + 1;

    dp = rx_frame(data_start:end);

    dp = reshape(dp(1:floor(length(dp)/sym_len)*sym_len), sym_len, []);
    dp = dp(n_guard+1:end,:);

    fd = fft(dp, n_fft, 1)/sqrt(n_fft);

    carrier = 4:126;
    rx_sc = fd(carrier,:);

    %% ---- 4.4 SNR估计（EVM）----
    SNR_frame(k) = 10*log10(mean(abs(rx_sc(:)).^2));

    fprintf("Frame %d SNR = %.2f dB\n", k, SNR_frame(k));

end

%% ================== 5. 汇总 ==================
valid = SNR_frame(SNR_frame > 0 & isfinite(SNR_frame));

if ~isempty(valid)
    SNR_mean = 10*log10(mean(10.^(valid/10)));
else
    SNR_mean = NaN;
end

fprintf("\n=============================\n");
fprintf("File SNR (mean) = %.2f dB\n", SNR_mean);
fprintf("=============================\n");