%作用：进行参数统一管理，支持 "结构体 / 零散参数" 两种输入方式，自动补全默认值、计算派生参数，输出规范的信号参数结构体；

function SIG = setSignalParams(varargin)

% Last Update: 08/01/2019


%% Input Parameters  处理 "不同输入方式"（代码的灵活度核心）
if nargin <= 3         % 模式1：输入是"结构体+可选采样率/采样点数"
    SIG = varargin{1}; % 第一个输入是核心结构体（比如包含symRate、M等）
    symRate = SIG.symRate;  % 从结构体提取符号率（必须有）

     % 从结构体提取可选参数（有就取，没有就先不定义）
    if isfield(SIG,'nBpS')
        nBpS = SIG.nBpS;
    end
    if isfield(SIG,'nPol')
        nPol = SIG.nPol;
    end
    if isfield(SIG,'rollOff')
        rollOff = SIG.rollOff;
    end


    % 补充采样率/采样点数（nargin=2→第二个输入是参数结构体；nargin=3→直接传采样率+点数
    if nargin == 2
        PARAM = varargin{2};
        sampRate = PARAM.sampRate;
        nSamples = PARAM.nSamples;
    elseif nargin == 3
        sampRate = varargin{2};
        nSamples = varargin{3};
    end



else % 模式2：输入是"参数名+参数值"成对传入（比如'symRate',10e9,'M',16）
    for n = 1:2:nargin % 步长2，逐个提取"名-值"对
        varName = varargin{n};   % 参数名（比如'symRate'）
        varValue = varargin{n+1}; % 参数值（比如10e9）
        switch varName
            % 匹配不同参数名，赋值给对应变量
            case {'symRate','symbol-rate'}
                symRate = varValue;
            case {'M'}
                SIG.M = varValue;  % 调制阶数（比如16=16QAM）
            case {'nBpS'}
                nBpS = varValue;
            case {'nPol'}
                nPol = varValue;
            case {'roll-off'}
                rollOff = varValue;
            case {'sampRate'}
                sampRate = varValue;
            case {'nSamples'}
                nSamples = varValue;
            case {'nSpS'}
                nSpS = varValue;
            case {'nSyms'}
                nSyms = varValue;   % 总符号数
            case 'encoding'
                SIG.encoding = varValue;
            case 'modulation'
                SIG.modulation = varValue;
        end
    end
end


% 必须有调制阶数M，否则报错
if ~isfield(SIG,'M')
    error('You must specify the constellation size, M');
end

% 补全默认参数（有就用用户的，没有就用默认值）
if ~exist('nBpS','var')
    nBpS = log2(SIG.M);
end   % nBpS默认=log2(M)（比如M=16→4）
if ~exist('nPol','var')
    nPol = 2;
end   % 偏振数默认=2（双偏振）
if ~exist('rollOff','var')
    rollOff = 0.05;
end  % 滚降系数默认=0.05
if ~isfield(SIG,'encoding')
    SIG.encoding = 'normal';
end  % 编码默认=普通
if ~isfield(SIG,'modulation')
    SIG.modulation = 'QAM';
end  % 调制方式默认=QAM

% 采样率补全：如果没传采样率，但传了nSpS→采样率= nSpS × 符号率
if ~exist('sampRate','var')
    if exist('nSpS','var')
        SIG.nSpS = nSpS;
        sampRate = nSpS * symRate;
    end
end

% 采样点数补全：如果没传采样点数，但传了总符号数→采样点数= (采样率/符号率) × 总符号数
if ~exist('nSamples','var')
    if exist('nSyms','var')
        SIG.nSyms = nSyms;
        if exist('sampRate','var')
            nSamples = sampRate/symRate * nSyms;
        end
    end
end

%% Secondary Parameters  计算派生参数（从基础参数算 "衍生参数"）
bitRate = symRate * nBpS * nPol;   % 比特率=符号率×每符号比特数×偏振数
tSym = 1/symRate;   % 符号周期=1/符号率（1个符号的持续时间）
tBit = nPol/bitRate;  % 比特周期=偏振数/比特率（单偏振的比特持续时间）


%% Signal Parameters that Depend on the Simulation Parameters计算和仿真相关的参数（采样率 / 点数相关）
%从 "仿真参数（采样率、采样点数）" 反推 "信号参数（总符号数、总比特数）"；
if exist('sampRate','var') && exist('nSamples','var')
    SIG.nSpS = sampRate / symRate;   % 每个符号的采样点数=采样率/符号率
%     SIG.nSpB = sampRate / (symRate * nBpS);
    SIG.nSyms = nSamples / SIG.nSpS;   % 每个符号的采样点数=采样率/符号率
    SIG.nBits = SIG.nSyms * nBpS;   % 总比特数=总符号数×每符号比特数
end

%% Set QAM fields  结构体SIG
SIG.symRate = symRate;
SIG.bitRate = bitRate;
SIG.nBpS = nBpS;
SIG.nPol = nPol;
SIG.tSym = tSym;
SIG.tBit = tBit;
SIG.rollOff = rollOff;

