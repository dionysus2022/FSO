%根据指定的目标输出信噪比（SNRout_dB），计算需要添加的噪声功率，
% 生成符合该功率的复高斯白噪声，叠加到原始信号S上，最终输出含噪信号，并返回噪声功率、实际输出信噪比等关键参数
function [S,Pn_Fs,SNR_out,noise] = setSNR(S,SNR,Fs,Rs)

% Last Update: 08/11/2019


%% Input Parser  输入参数适配 
%兼容两种输入方式：
% 直接传数字（SNR=10）：自动封装成结构体,默认为SNRout-dB=10 
% 传结构体（如SNR.SNRout_dB=10, SNR.SNRin_dB=5）：直接使用
if isnumeric(SNR)
    SNR_tmp.SNRout_dB = SNR;  % 如果SNR是纯数字（比如10），转成结构体
    SNR = SNR_tmp;
end
if ~isfield(SNR,'SNRin_dB')
    SNR.SNRin_dB = Inf;  % 默认输入信噪比为无穷大（无输入噪声）
end

%% Input Parameters
[nPol,nSamples] = size(S);  % 信号维度：nPol=偏振数，nSamples=采样点数
SNRout = 10.^(SNR.SNRout_dB/10);%从dB形式化成数值形式，为15.84
SNRin = 10^(SNR.SNRin_dB/10); % 输入信噪比转线性值（Inf→无穷大）

%% Determine Signal Power  计算信号功率
%复信号功率计算：abs(S)^2是复信号的瞬时功率（实部 ²+ 虚部 ²），mean(...,2)表示 "按行（偏振）求均值"；
Ps_in = mean(abs(S).^2,2);% 输入的信号能量
if isfield(SNR,'Pin')
    Ps = SNR.Pin;   % 用户指定信号功率（比如标准化为1）
else
    Ps = Ps_in;   % 用信号实际功率
end

%% Determine Noise Power 计算噪声功率
if isfield(SNR,'Pn')
    % 场景1：用户直接指定噪声功率（Pn_Fs）
    Pn_Fs = SNR.Pn;
    if numel(Pn_Fs) == 1
        Pn_Fs = repmat(Pn_Fs,1,nPol); % 扩展到所有偏振
    end
    Pn_Rs = Pn_Fs*Rs/Fs;  % 采样率带宽→符号率带宽的噪声功率换算
else
    % 场景2：按目标SNR计算噪声功率（默认场景）
    Pn0 = Ps/SNRin;   % 输入噪声功率（SNRin=Inf→Pn0=0）
    Pn_Rs = Ps./SNRout - Pn0;%平均信号能量除以信噪比等于噪声能量
    Pn_Fs = Pn_Rs*Fs/Rs;  % 换算到采样率带宽（用于生成噪声）
end

%% Generate Noise
% Set random number generator seed:设置随机种子（保证可复现）
if isfield(SNR,'noiseSeed')  % 用户指定种子→噪声可复现
    rng(SNR.noiseSeed);
else
    rng('shuffle');%根据当前时间初始化生成器，在每次调用 rng 后会产生一个不同的随机数序列。
    tmp = rng;
    SNR.noiseSeed = tmp.Seed;  % 保存种子，方便后续复现
end
% Generate noise in the I and Q components: % 生成I/Q两路噪声（实部/虚部）
[noise_I,noise_Q] = deal(zeros(nPol,nSamples));
for n = 1:nPol
    % randn生成标准正态噪声（均值0，方差1），乘以sqrt(Pn_Fs/2)缩放功率
    noise_I(n,:) = randn(1,nSamples).*sqrt(Pn_Fs(n)/2);%这里randn就是random normal符合正态的噪声。
    noise_Q(n,:) = randn(1,nSamples).*sqrt(Pn_Fs(n)/2);
end
% Create the complex-valued noise:
% 合成复噪声（I=实部，Q=虚部）
noise = noise_I + 1j*noise_Q;

%% Add Noise to the Signal  叠加噪声到信号
S = S + noise;  % 逐点叠加复噪声，得到含噪信号

%% Calculate Ouput SNR  计算实际输出SNR
Pn_Fs = mean(Pn_Fs);   % 求所有偏振的平均噪声功率
SNR_out = pow2db(mean(Ps_in)/mean(Pn_Rs));   % 线性值转dB，验证实际SNR
