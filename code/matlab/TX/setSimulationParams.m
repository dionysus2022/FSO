%作用：根据输入的采样率（sampRate） 和总采样点数（nSamples），计算仿真所需的所有基础时间 / 频率参数
function PARAM = setSimulationParams(sampRate,nSamples)

% Last Update: 08/08/2019


%% Secondary Parameters 计算核心派生参数
% 1. 计算信号总时长（时间窗口长度）
tWindow = nSamples / sampRate;%时间窗口的长度，采样点数除以采样率得花多少时间
% 2. 计算采样间隔（每个样本点的时间差）
dt = 1 / sampRate; %采样率倒数是采样时间，每个符号多少时间
% 3. 计算频率分辨率（频谱分析的最小频率间隔）
df = sampRate / nSamples;% 采完整个262144符号的整体频率
% 4. 生成时间轴（每个样本点对应的具体时刻）
t = (0:nSamples-1)*dt;%采样所需总时间
% 5. 生成频率轴（对称的频率范围，适配FFT频谱分析）
f = (-nSamples/2:nSamples/2-1)*(sampRate/nSamples);

%% Set PARAM fields  把所有参数存入结构体（最终输出）
PARAM.sampRate = sampRate;%采样率
PARAM.nSamples = nSamples;%总采样点数
PARAM.tWindow = tWindow;%信号总时长
PARAM.df = df;%频率采样率
PARAM.dt = dt;%采样间隔
PARAM.t = t;%时间轴
PARAM.f = f;%频率轴

