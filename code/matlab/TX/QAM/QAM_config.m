%QAM调制星座图的核心配置函数
%作用：一键配置QAM的所有关键参数，调制阶数、偏振态数、编码方式等参数，
%输出包括一个星座图(IQmap)、比特-符号映射（sym2bitMap）、平均功率。调制合适
function QAM = QAM_config(varargin)
%QAM_config     Configure QAM struct based on a selected format 
%   This functions configures a struct with QAM properties based on the
%   specified modulation format. Currently supported formats are:
%   QPSK/4QAM (square)
%   8QAM (cross)
%   16QAM (square)
%   32QAM (cross)
%   36QAM (square)
%   64QAM (square)
%   128QAM (cross)
%   256QAM (square)
%   512QAM (cross)
%   1024QAM (square)
%
%   INPUTS:
%   modFormat   :=  modulation format string. It must contain the 'mQAM'
%                   string in it (e.g. 'QPSK','4QAM','8QAM','1024QAM') and
%                   it may also contain a string to indicate the use of
%                   dual-polarization (either 'DP' or 'PM')
%   encoding    :=  method for symbol encoding
%                   - 'normal': normal encoding (non-differential) (defaul)
%                   - 'diff-quad': differential quadrant encoding
%
%   OUTPUTS:
%   QAM         :=  struct with QAM parameters
%
%
%   Examples:
%       QAM = QAM_config('1024QAM')
%       QAM = QAM_config('PM-QPSK')
%       QAM = QAM_config('DP-4QAM')
%       QAM = QAM_config('PM-32QAM')
%
%       scatterplot(QAM.IQmap);
%
%
%   Author: Fernando Guiomar
%   Last Update: 10/04/2019

% 输入：支持两种输入方式（结构体 / 键值对），核心参数包括M（调制阶数，如 4/16/64）、nPol（偏振态数，1/2）、encoding（编码方式）、modulation（调制类型，QAM/PAM/PSK）等；
% 输出：QAM结构体（包含调制格式、星座图、比特映射、功率等所有关键参数）；
% 核心价值：把复杂的 QAM 星座图生成、参数计算封装成一个函数，无需手动计算星座点坐标、比特映射，直接调用即可得到标准化的调制参数。

%% Input Parser 输入参数解析，“参数接收模块，支持两种输入模式，同时设置默认值，
% Default parameter values:
nPol = 2;  % 默认双偏振
encoding = 'normal';% 默认普通编码（非差分）
modulation = 'QAM'; % 默认调制类型为QAM
class = [];% QAM星座类型（square/cross，默认自动判断）
symbolOrder = 'gray'; % 默认格雷编码（通信中抗误码的最优编码）

%第二步：处理两种输入方式
% Assignment of input parameters:
if nargin == 1                   % 输入方式1：单个结构体（如SIG.M=16, SIG.nPol=2）
    SIG = varargin{1};
    M = SIG.M;
     % 逐个读取结构体中的参数，有则覆盖默认值，无则保留默认
    if isfield(SIG,'nPol')
        nPol = SIG.nPol;
    end
    if isfield(SIG,'encoding')
        encoding = SIG.encoding;
    end
    if isfield(SIG,'modulation')
        modulation = SIG.modulation;
    end
    if isfield(SIG,'class')
        class = SIG.class;
    end
    if isfield(SIG,'symbolOrder')
        symbolOrder = SIG.symbolOrder;
    end
else                     % 输入方式2：键值对（如'M',16,'nPol',2）
    for n = 1:2:nargin          % 步长2，依次读取"参数名-参数值"
        varName = varargin{n};
        varValue = varargin{n+1};
        if strcmpi(varName,'M')
            M = varValue;
        elseif strcmpi(varName,'nPol')
            nPol = varValue;
        elseif strcmpi(varName,'encoding')
            encoding = varValue;
        elseif strcmpi(varName,'modulation')
            modulation = varValue;
        elseif strcmpi(varName,'class')
            class = varValue;
        elseif strcmpi(varName,'symbolOrder')
            symbolOrder = varValue;
        end
    end
end

% 第三步：自动判断QAM星座类型（square/cross）
if strcmp(modulation,'QAM') && isempty(class)
    if mod(sqrt(M),1) == 0
        class = 'square';% 平方根是整数（如16QAM→sqrt(16)=4）→square（方形星座）
    else
        class = 'cross'; % 平方根非整数（如8QAM→sqrt(8)≈2.828）→cross（十字形星座）
    end
end



%% Assign Constellation  生成QAM星座图，优先用 MATLAB 内置函数生成星座图，失败则调用自定义函数加载预制星座图
try  % 尝试用MATLAB内置函数生成星座图
    symbolMap = 0:M-1;            % 符号索引（0到M-1，对应M个星座点）
    switch modulation
        case 'QAM'
            if M == 8        % 8QAM特殊处理（无十字形8QAM）
                if strcmp(class,'cross')
                    error('Cannot generate cross-8QAM with qammod.');
                elseif isempty(class)
                    class = 'rect';
                    M_rect = [2 4];   % 矩形8QAM
                end
            end
             % 生成QAM星座图：qammod(符号索引, 阶数, 编码方式)
            const = qammod(symbolMap,M,symbolOrder);
        case 'PAM'   %脉冲幅度调制（备用）
            const = pammod(symbolMap,M,pi/2,symbolOrder);
        case 'PSK'    % 相移键控（如QPSK属于4PSK）
            const = pskmod(symbolMap,M,0,symbolOrder);
    end
catch   % 内置函数生成失败（如十字形32QAM），加载预制星座图
    % Load QAM Constellation:
    MF_ID = [num2str(M) modulation '_' class];   % 生成星座图ID（如32QAM_cross）
    [const,symbolMap] = QAM_loadConstellation(MF_ID);  % 调用自定义函数加载
end

%% Configure Modulation Format Parameters    计算调制关键参数（星座图特征提取）
% Determine all radii in the constellation:    这部分是 "参数加工"，从生成的星座图中提取功率、半径、比特 - 符号映射等关键参数。
%1. 计算星座图的半径（归一化）
radius = unique(abs(const));                           % 提取所有不同的星座点半径（模长）
radius = sort(radius/max(radius),2,'descend');          % 归一化+降序排序
% 2. 生成调制格式名称（如PM-16QAM，PM=双偏振）
modFormat = [num2str(M) modulation];
% If there are two polarization, change modulation format ID accordingly:
if nPol == 2
    modFormat = ['PM-' modFormat];
end

%% Determine Constellation Mapping   3. 构建星座点索引映射（保证符号顺序正确）
% Symbol mapping and indices:
symbolInd = zeros(1,M);
for n = 0:M-1
    symbolInd(n+1) = find(symbolMap==n);   % 找到每个符号对应的星座点位置
end
IQmap = const(symbolInd).';         % 最终的IQ映射矩阵（星座图）


% Mapping symbols to bits:   % 4. 构建比特-符号映射表（sym2bitMap）
if mod(log2(M),1) == 0         % M是2的整数次幂（所有QAM都满足）
    sym2bitMap = false(M,log2(M));    % M行×每符号比特数列的逻辑矩阵
    for n = 0:M-1
        [~,e] = log2(n);
         % 核心：把符号索引n转换为二进制比特（如n=5→101）
        sym2bitMap(n+1,:) = rem(floor(n * pow2(1-max(e,log2(M)):0)),2);
    end
end

% LSB bit map for differential encoding:
if strcmp(encoding,'diff-quad')
    QAM.LSB_bitMap = NaN(M,log2(M)-2);
end
% 5. 计算星座图的平均功率和最大功率（归一化用）
% Calculate average and maximum constellation powers:
S_meanP = mean(abs(IQmap).^2);         % 平均功率：所有星座点模长平方的均值
S_maxP = max(abs(IQmap).^2);           % 最大功率：星座点模长平方的最大值



%% Output QAM Struct   封装输出结构体QAM
QAM.modFormat = modFormat;                                                  % modulation format 调制格式名称
QAM.mode = modulation;                                                      % modulation type (QAM, PAM, PSK)  调制类型（QAM PSK PAM）
if strcmp(QAM.mode,'QAM')
    QAM.class = class;                                                      % QAM class (square, cross, rect) QAM星座类型（square/cross）
    if strcmp(QAM.class,'rect')
        QAM.M_rect = M_rect;
    end
end
QAM.M = M;                                                                 % constellation number of symbols 调制阶数 
QAM.nBpS = log2(M);                                                        % number of bits per symbol  每符号比特数    
QAM.entropy = log2(M);                                                     %熵（和每符号比特数一致）
QAM.radius = radius;                                                       % constellation radius 星座图半径（归一化） 
QAM.nPol = nPol;                                                           % number of polarization components    偏振态数
QAM.meanConstPower = S_meanP;                                               % mean constellation power         星座平均功率 
QAM.maxConstPower = S_maxP;                                                 % max constellation power  星座最大功率
QAM.IQmap = IQmap;                                                          % mapping between constellation symbols and IQ  核心：IQ星座图映射
if exist('sym2bitMap','var')
    QAM.sym2bitMap = sym2bitMap;                                            % map symbols to bits  比特-符号映射表
end
QAM.encoding = encoding;   %编码方式
