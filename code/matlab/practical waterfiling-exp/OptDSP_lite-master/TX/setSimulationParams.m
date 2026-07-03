function PARAM = setSimulationParams(sampRate,nSamples)

% Last Update: 08/08/2019


%% Secondary Parameters
tWindow = nSamples / sampRate;%时间窗口的长度，采样点数除以采样率得花多少时间
dt = 1 / sampRate; %采样率倒数是采样时间，每个符号多少时间
df = sampRate / nSamples;% 采完整个262144符号的整体频率
t = (0:nSamples-1)*dt;%采样所需总时间
f = (-nSamples/2:nSamples/2-1)*(sampRate/nSamples);

%% Set PARAM fields
PARAM.sampRate = sampRate;
PARAM.nSamples = nSamples;
PARAM.tWindow = tWindow;
PARAM.df = df;
PARAM.dt = dt;
PARAM.t = t;
PARAM.f = f;

