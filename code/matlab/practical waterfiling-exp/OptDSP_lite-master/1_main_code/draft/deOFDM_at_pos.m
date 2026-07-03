function [SRX_deofdm] = deOFDM_at_pos(SRX, ofdm, sizeofqamsymbol, frame_start_idx)
% deOFDM_at_pos - 跳过 packet_edge_power_dect 粗同步，直接从指定位置开始解调
% 输入:
%   SRX             - 行向量，接收信号（已归一化）
%   ofdm            - OFDM 参数结构体
%   sizeofqamsymbol - 每帧 OFDM 数据符号数 (SIG.nSyms)
%   frame_start_idx - 帧起始索引（preamble 80个0的起始位置，从1开始）
%
% 与 deOFDM 的区别：
%   1. 跳过 packet_edge_power_dect（粗同步）
%   2. 用局部互相关精确定位长训练序列（LTR）位置
%   3. 搜索范围限制在期望位置 ±50 样本，避免多帧干扰

Ltrs_num      = 2;
zeros_head    = 80;    % 帧头0的个数
plot_en       = 0;     % 关闭绘图
Fg            = 1e9;   % 信号带宽
LtrsCPLength  = ofdm.NumberOfGuardTime;

NumberOfIFFTSamples   = ofdm.NumberOfIFFTSamples;
Carrier_location      = ofdm.Carrier_location_demo;
NumberOfCarriers      = length(ofdm.Carrier_location_demo);
NumberOfGuardTime     = ofdm.NumberOfGuardTime;
NrOfOFDMSymbols       = ofdm.size;

% 计算帧长
symbol_bits = zeros_head + NumberOfGuardTime + NumberOfIFFTSamples * Ltrs_num ...
              + (NumberOfIFFTSamples + NumberOfGuardTime) * sizeofqamsymbol;

rx_signal_recovered = SRX;

%% 生成长训练序列
load 'LongTrainSym_ini.mat'
LongTrainSym = LongTrainSym_ini(1:NumberOfIFFTSamples);
LongTrainSym([1 129]) = 0;
ltrs_ifft_in = LongTrainSym;
ltrs_ifft_in(1, NumberOfIFFTSamples/2+2:NumberOfIFFTSamples) = conj(ltrs_ifft_in(1, NumberOfIFFTSamples/2:-1:2));
LongTrainSym = ltrs_ifft_in;

%% 局部互相关精确定位第一个长训练序列（LTR）的 IFFT 输出起始位置
% 帧结构: [80 zeros][CP(16)][IFFT(256)_LTR1][CP(16)][IFFT(256)_LTR2][data...]
% LTR1 IFFT 输出在帧内位置: zeros_head + LtrsCPLength + 1 = 97
ifft_LTRS = ifft(LongTrainSym);  % 1 x 256

expected_ltrs_pos = frame_start_idx + zeros_head + LtrsCPLength; % 期望的 LTR1 IFFT 起始位置

% 搜索窗口: 期望位置 ±50 样本
search_start = max(1, expected_ltrs_pos - 50);
search_end   = min(length(rx_signal_recovered) - length(ifft_LTRS), expected_ltrs_pos + 50);

best_corr = 0;
fine_index_1 = expected_ltrs_pos;  % 默认值

for idx = search_start : search_end
    seg = rx_signal_recovered(idx : idx + length(ifft_LTRS) - 1);
    corr_val = abs(sum(seg .* conj(ifft_LTRS)));
    if corr_val > best_corr
        best_corr = corr_val;
        fine_index_1 = idx;
    end
end

%% 提取一帧数据（与 deOFDM 原始逻辑一致）
% rx_signal 从 LTR 起始位置前 (zeros_head + LtrsCPLength - 5) = 91 个样本开始
rx_start = fine_index_1 - (zeros_head + LtrsCPLength - 5);
rx_end   = fine_index_1 + symbol_bits - 1;

% 处理信号边界：超出范围时补零
if rx_start < 1
    rx_signal = [zeros(1, 1 - rx_start), rx_signal_recovered(1 : min(rx_end, length(rx_signal_recovered)))];
elseif rx_end > length(rx_signal_recovered)
    rx_signal = [rx_signal_recovered(rx_start : length(rx_signal_recovered)), zeros(1, rx_end - length(rx_signal_recovered))];
else
    rx_signal = rx_signal_recovered(rx_start : rx_end);
end

% 精细同步：在 rx_signal 中找 LTR 位置
% LTR 应在 rx_signal 的第 (zeros_head + LtrsCPLength - 4) ≈ 92 个样本处
% 使用已知偏移量，避免 rx_fine_time_sync_cross_corr 的宽搜索范围找到错误峰值
fine_time_est = fine_index_1 - rx_start + 1;  % LTR 在 rx_signal 中的精确位置

syn2_time_signal = rx_signal(fine_time_est : length(rx_signal));

%% 频偏估计
[freq_offset, syn3_signal] = freq_offset_esti(syn2_time_signal, Fg, NumberOfIFFTSamples, Ltrs_num);

%% FFT 还原数据
[freq_tr_syms, freq_data_syms] = rx_FFT(syn2_time_signal, NumberOfIFFTSamples, ...
    NumberOfGuardTime, sizeofqamsymbol, Ltrs_num);

SRX = freq_data_syms;
SignalTimeIX = real(SRX);
SignalTimeQX = imag(SRX);

SignalBaseIX = reshape(SignalTimeIX, NumberOfIFFTSamples + NumberOfGuardTime, NrOfOFDMSymbols);
SignalBaseQX = reshape(SignalTimeQX, NumberOfIFFTSamples + NumberOfGuardTime, NrOfOFDMSymbols);

%% Remove cyclic prefix
SignalBaseIX = SignalBaseIX(NumberOfGuardTime + 1 : NumberOfIFFTSamples + NumberOfGuardTime, :);
SignalBaseQX = SignalBaseQX(NumberOfGuardTime + 1 : NumberOfIFFTSamples + NumberOfGuardTime, :);

%% FFT
SpectralIX = fft(SignalBaseIX) ./ sqrt(NumberOfIFFTSamples);
SpectralQX = fft(SignalBaseQX) ./ sqrt(NumberOfIFFTSamples);
SpectralIQX = SpectralIX + 1i .* SpectralQX;

%% Carrier location
SpectralrcX = SpectralIQX.';

%% 信道估计与均衡
channel_est = channel_esti(freq_tr_syms, [1 129], LongTrainSym, Ltrs_num);
freq_data_syms = rx_channel_equal(SpectralrcX, channel_est, sizeofqamsymbol);

%% 提取数据子载波
IQXmatrix = zeros(NrOfOFDMSymbols, NumberOfCarriers);
for i = 1:NrOfOFDMSymbols
    IQXmatrix(i, :) = freq_data_syms(i, Carrier_location);
end

%% Normalization
SRX_deofdm = reshape(IQXmatrix, 1, size(IQXmatrix, 1) * size(IQXmatrix, 2));

end
