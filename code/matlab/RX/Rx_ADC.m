%核心功能：接收端 ADC（模数转换器）的仿真逻辑，整体功能是对接收信号Srx依次施加 ADC 的非理想特性（偏斜、低通滤波、重采样、限幅、PAPR 限制、量化）
function [Srx,ADC,Fs] = Rx_ADC(Srx,ADC,Fs,Rs)

% Last Update: 09/10/2019


%% Introduce Rx Skew 引入接收端IQ偏斜——模拟ADC的IQ之路不同步
if isfield(ADC,'SKEW') %skew IQ偏斜
    Srx = DAC_IQskew(Srx,Fs,ADC.SKEW);
end

%% Low-Pass Filtering  低通滤波——模拟ADC前端的抗混叠滤波器
if isfield(ADC,'LPF') %LPF低通滤波
    if nargin == 3
        Srx = LPF_apply(Srx,ADC.LPF,Fs);
    elseif nargin == 4
        Srx = LPF_apply(Srx,ADC.LPF,Fs,Rs);
    end
end

%% Resample to ADC Sampling Rate  重采样到ADC采样率——模拟ADC的采样率转换
if isfield(ADC,'RESAMP')      %将输入信号的采样率转换为ADC的实际采样率
    if isfield(ADC.RESAMP,'sampRate')
        Fs_ADC = ADC.RESAMP.sampRate;  % 直接指定ADC采样率
    elseif isfield(ADC.RESAMP,'nSpS') && nargin==4
        Fs_ADC = ADC.RESAMP.nSpS * Rs;  % 按每符号采样点数计算ADC采样率
    end
    Srx = applyResample(Srx,Fs,Fs_ADC);   % 重采样
    Fs = Fs_ADC;    % 更新采样率为ADC采样率
end

%% Apply Clipping   普通限幅——模拟ADC的最大输入幅度限制
if isfield(ADC,'clipping')
    Srx = DAC_applyClipping(Srx,ADC.clipping);
end

%% Set Maximum PAPR  限制最大PAPR——模拟ADC对峰值功率的约束
if isfield(ADC,'maxPAPR_dB') && ~isinf(ADC.maxPAPR_dB)
    [Srx,ADC.clip_I,ADC.clip_Q] = DAC_setMaxPAPR(Srx,ADC.maxPAPR_dB);
end

%% Quantization  量化——模拟ADC的模数转换（把连续的模拟信号转离散数字信号）
if isfield(ADC,'ENOB') && ~isinf(ADC.ENOB)  
    Srx = setENOB(Srx,ADC.ENOB);  % 按有效位数（ENOB）量化
elseif isfield(ADC,'nBits') && ~isinf(ADC.nBits)
    Srx = quantizeSignal(Srx,ADC.nBits);   % 按比特数（nBits）量化
end
end

