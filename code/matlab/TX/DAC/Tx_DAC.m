%输入参数 stx-理想发射信号（I/Q 两路） DAC-DAC的配置参数（模拟各种硬件缺陷） Fs-采样率
%输出：Stx - 处理后的信号（添加了各种损伤）  DAC - 可能更新过的DAC参数  PARAM - 重采样相关参数

function [Stx,DAC,PARAM] = Tx_DAC(Stx,DAC,Fs)

% Last Update: 09/10/2019


%% Resample to DAC Sampling Rate 重采样
if isfield(DAC,'RESAMP')
    [Stx,PARAM] = applyResample(Stx,Fs,DAC.RESAMP.sampRate);%重采样的目的是改变采样率，即将输入进来的采样率为FS改变为期望的采样率DAC.RESAMP.sampRate，这里为相等
end
                                                           %这个程序只执行了这一节，下面都没执行
%% Introduce Tx Skew  引入I/Q时延
if isfield(DAC,'SKEW') && isfield(DAC.SKEW,'delay_ps')
    Stx = DAC_IQskew(Stx,Fs,DAC.SKEW.delay_ps);
end

%% Insert Clock Timing Offset  时钟偏移
if isfield(DAC,'CLOCK') && isfield(DAC.CLOCK,'offset')
    [Stx,DAC.CLOCK] = DAC_clockOffset(Stx,Fs,DAC.CLOCK);
    %发射端和接收端的时钟频率不完全一致，会导致相位噪声和符号定时误差
end

%% Insert Clock Timing Jitter  时钟抖动
if isfield(DAC,'CLOCK')
    [Stx,DAC.CLOCK] = DAC_clockJitter(Stx,Fs,DAC.CLOCK);
    %时钟信号的随机微小变化，会增加噪声，降低信噪比
end

%% Apply Clipping  信号削波
if isfield(DAC,'clipping')
    Stx = DAC_applyClipping(Stx,DAC.clipping);
    %信号幅度超过DAC的最大可输出范围，会消掉超出阈值的部分，但会使信号失真，产生谐波
end

%% Set Maximum PAPR 限制峰均功率比
if isfield(DAC,'maxPAPR_dB') && ~isinf(DAC.maxPAPR_dB)
    [Stx,DAC.clip_I,DAC.clip_Q] = DAC_setMaxPAPR(Stx,DAC.maxPAPR_dB);
    %PAPR：峰均功率比  目的：防止信号峰值功率过高
end

%% Quantization 量化
if isfield(DAC,'ENOB') && ~isinf(DAC.ENOB)
    Stx = setENOB(Stx,DAC.ENOB); %按有效位数量化
elseif isfield(DAC,'nBits') && ~isinf(DAC.nBits)
    Stx = quantizeSignal(Stx,DAC.nBits); %按总位数量化
end

%% Low-Pass Filtering  低通滤波 滤除高频噪声和镜像分量
if isfield(DAC,'LPF')
    Stx = LPF_apply(Stx,DAC.LPF,Fs);
end

%% Normalization  归一化——调整信号幅度到DAC的可接受范围
if isfield(DAC,'NORM')
    Stx = normalizeIQ(Stx,DAC.NORM.range,DAC.NORM.mode);
end
