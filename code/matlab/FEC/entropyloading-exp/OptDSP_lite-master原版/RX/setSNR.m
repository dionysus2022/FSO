function [S,Pn_Fs,SNR_out,noise] = setSNR(S,SNR,Fs,Rs)

% Last Update: 08/11/2019


%% Input Parser
if isnumeric(SNR)
    SNR_tmp.SNRout_dB = SNR;
    SNR = SNR_tmp;
end
if ~isfield(SNR,'SNRin_dB')
    SNR.SNRin_dB = Inf;
end

%% Input Parameters
[nPol,nSamples] = size(S);
SNRout = 10.^(SNR.SNRout_dB/10);%从dB形式化成数值形式，为15.84
SNRin = 10^(SNR.SNRin_dB/10);

%% Determine Signal Power
Ps_in = mean(abs(S).^2,2);% 输入的信号能量,明显这个是会变的，因为输入星座点数据是随机的，是实际序列求出来的能量
if isfield(SNR,'Pin')
    Ps = SNR.Pin;
else
    Ps = Ps_in;
end

%% Determine Noise Power
if isfield(SNR,'Pn')
    Pn_Fs = SNR.Pn;
    if numel(Pn_Fs) == 1
        Pn_Fs = repmat(Pn_Fs,1,nPol);
    end
    Pn_Rs = Pn_Fs*Rs/Fs;
else
    Pn0 = Ps/SNRin;
    Pn_Rs = Ps./SNRout - Pn0;%平均信号能量除以信噪比等于噪声能量
    Pn_Fs = Pn_Rs*Fs/Rs;
end

%% Generate Noise
% Set random number generator seed:
if isfield(SNR,'noiseSeed')
    rng(SNR.noiseSeed);
else
    rng('shuffle');%根据当前时间初始化生成器，在每次调用 rng 后会产生一个不同的随机数序列。
    tmp = rng;
    SNR.noiseSeed = tmp.Seed;
end
% Generate noise in the I and Q components:
[noise_I,noise_Q] = deal(zeros(nPol,nSamples));
for n = 1:nPol
    noise_I(n,:) = randn(1,nSamples).*sqrt(Pn_Fs(n)/2);%这里randn就是random normal符合正态的噪声。为什么可以这样直接乘？因为这里高斯符合N(0,1)也就是均值为零，
    noise_Q(n,:) = randn(1,nSamples).*sqrt(Pn_Fs(n)/2);%方差为1，而方差Σ(xi-μ)^2/n,这里均值μ为零，刚好方差等于平均能量，既然平均能量为1，直接乘噪声能量不就是这个噪声的能量吗，分到IQ路自然就开个根号
end                                                                                %需要强调的是，上面得到的信号能量是实际信号的平均能量，而这里噪声能量是平均能量，而不是产生的所有加和的噪声一定满足信噪比。关键在于先后问题，信号是先产生后算能量的，因为每个星座点概率不一样，所以不会为普通QAM的平均能量。而噪声是先算能量，后随机产生的，产生出来的噪声去算能量，未必一定是符合信噪比要求的。所以点数越多，接收端计算EVM愈接近信噪比要求
% Create the complex-valued noise:
% 另外，为什么是Pn_Fs而不是Pn_Rs除以2（除以2是因为能量分了IQ两路）？这是因为后面的下采样要删掉一半的点，而这些点也是带着一半的噪声能量的，直接删掉的话这边自然要先翻个倍
noise = noise_I + 1j*noise_Q;
noisemean = mean(abs(noise).^2,2);               %计算产生的噪声的能量均值，其实到这一步，会发现信噪比是翻了两倍（每个符号采样2的原因）的，因为是Pn_Fs而不是Pn_Rs的原因
%% Add Noise to the Signal
S = S + noise;
%% Calculate Ouput SNR
Pn_Fs = mean(Pn_Fs);
SNR_out = pow2db(mean(Ps_in)/mean(Pn_Rs));
