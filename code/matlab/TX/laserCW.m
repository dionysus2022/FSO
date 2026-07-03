%作用：生成连续波（CW）激光器光场信号"，生成相位噪声、强度噪声；
% 模拟真实激光器发射的光场信号—— 不仅生成理想的恒定功率 / 相位光信号，还加入了激光器的两个核心噪声（相位噪声、强度噪声）
% 输入：
% LASER：激光器参数结构体（包含线宽、噪声、初始相位、功率等，缺省参数会自动补）；
% Fs：采样率 [Hz]（比如 1e9，即 1GHz）；
% nSamples：需要生成的信号采样点数（比如 10000）；
% 输出：
% A：激光器输出的光场复振幅（核心输出，包含幅度 / 相位信息）；
% LASER：更新后的结构体
function [A,LASER] = laserCW(LASER,Fs,nSamples)

% Last Update: 07/08/2019


%% Input Parser
if ~isfield(LASER,'linewidth')
    LASER.linewidth = 0;% 默认无相位噪声（理想激光器）
end
if ~isfield(LASER,'RIN_dB')
    LASER.RIN_dB = -inf;  %dB为负无穷说明相对强度噪声为0
end
if ~isfield(LASER,'phase0')
    LASER.phase0 = 0;   % 默认初始相位为0弧度
end
if ~isfield(LASER,'P0_dBm')
    LASER.P0_dBm = 30;      % 1 Watt (per polarization)   默认功率30dBm（等于1瓦，光通信常用值）
end

%% Input Parameters  %参数提取（把结构体参数 "拎出来方便用"）
% Laser parameters:
lw = LASER.linewidth;       % laser linewidth [Hz]  激光器线宽 [Hz]（线宽越大，相位噪声越严重）
RIN_dB = LASER.RIN_dB;      % relative intensity noise [dB/Hz]  相对强度噪声（RIN）[dB/Hz]（负数越小，噪声越小）
ph0 = LASER.phase0;         % laser initial phase [rad]   初始相位 [弧度]
P0_dBm = LASER.P0_dBm;      % laser emitted power [dBm]  输出功率 [dBm]（光通信常用功率单位）



%% LASER phase noise   生成相位噪声（模拟激光 "频率 / 相位的随机波动"），
%  激光的相位不是恒定的，而是随时间随机漂移（就像钟摆的摆动角度随机偏移）：
phVar = 2*pi*lw/Fs; % 相位噪声的方差（核心公式，记结论即可）
phNoise = sqrt(phVar)*randn(1,nSamples);  % 生成高斯随机相位噪声
phNoise = cumsum(phNoise);  % 累加→相位噪声是"累积的随机游走"


%% LASER intensity noise 生成强度噪声（模拟激光 "功率的随机波动"），激光的功率不是恒定的，而是有微小波动
P0 = db2pow(P0_dBm-30); % dBm转成实际功率（瓦）：db2pow(x)=10^(x/10)
intVar = 10^(RIN_dB/10)*Fs*P0^2;  % 强度噪声的方差（核心公式）                 
intNoise = sqrt(intVar)*randn(1,nSamples); % 生成高斯强度噪声

%% LASER transmitted optical field  生成最终的光场复振幅（核心输出）
A = sqrt(P0 + intNoise) .* exp(1j*(ph0 + phNoise));
%光场是复数值（幅度 + 相位），符合通信里 "复基带信号" 的表示方式；
% sqrt(P0 + intNoise)：光场的幅度—— 由 "平均功率 P0 + 强度噪声 intNoise" 开平方得到（因为光场幅度∝√功率）；
% exp(1j*(ph0 + phNoise))：光场的相位—— 由 "初始相位 ph0 + 相位噪声 phNoise" 决定（复数的指数形式表示相位）；
% .*：逐元素相乘（Matlab 里的点乘，保证每个采样点的幅度 × 相位对应）；


%% Output LASER Struct  把生成的噪声、方差等参数回写到LASER结构体里
LASER.phaseVar = phVar;         % phase variance相位噪声方差
LASER.phaseNoise = phNoise;     % phase noise [rad]相位噪声值【弧度】
LASER.RIN_dB = RIN_dB;          % relative intensity noise [dB/Hz] 相位强度噪声
LASER.intNoise = intNoise;      % intensity noise [W]强度噪声值
