%核心功能概率整形 QAM（PS-QAM），让星座图上 "距离原点近、功率小" 的点被更多使用，"距离远、功率大" 的点少用，从而降低整体传输功率，提升通信系统的能效。
%输入是二进制比特流 + 基础 QAM 参数，
% 输出是经过概率整形的 QAM 调制信号，同时计算并保存整形后的关键参数（比如星座点概率、平均功率等）。
function [Stx,txSyms,QAM] = Tx_ProbShaping(txBits,QAM,SIG,R_FEC)

% Last Update: 15/10/2019


%% Input Parser
% Set Default Shaping Method:
%第一部分：检查QAM结构体里有没有 "概率整形方法（PCS）"，
% 如果没有，默认用CCDM（一种常用的分布匹配算法，不用纠结名字，记 "是概率整形的核心算法" 就行）；
if ~isfield(QAM,'PCS') || ~isfield(QAM.PCS,'method')
    QAM.PCS.method = 'CCDM';
end
% Set Default FEC Rate:
%第二部分：检查输入参数R_FEC（FEC 编码率）是否传入 / 为空，没传就默认设为 1（即不做 FEC 编码）；
if nargin < 4 || isempty(R_FEC)
    R_FEC = 1;
end

%% Impact of FEC on the PAS Scheme  FEC 对概率整形的约束计算（核心前提：确定整形的 "规则边界"）
nBpS = SIG.nBpS;  % 从系统参数取"每符号目标比特数"
C = QAM.IQmap;  % 原始星座图映射表（比如64QAM的IQ坐标）
M_PS = numel(C); % 星座点数量（比如64QAM的M_PS=64）
flag = false;  % 标记：是否需要扩展星座图（默认不需要）

% 检查星座点数量是否是2的整数次幂（QAM要求星座数是2^k）
if mod(log2(M_PS),1)
    M_PS = 2^nextpow2(M_PS); % 扩展到最近的2的幂次（比如63→64，70→128）
    if mod(sqrt(M_PS),1)     % 扩展到最近的2的幂次（比如63→64，70→128）
        M_PS = 2^nextpow2(M_PS+1);
    end
    flag = true;  % 标记：星座图被扩展了
end

% 计算FEC编码占用的"熵开销"（核心公式，先记结论）
H_quad = (1-R_FEC)*log2(M_PS);%FEC的熵的开销
% 计算允许的最小FEC编码率（防止熵开销超标）
R_FEC_min = (log2(M_PS) - 2) / log2(M_PS);

% 关键校验：FEC的熵开销不能超过2比特/符号（四象限的最大熵）
if H_quad > 2
    error(['PAS scheme requires to allocate FEC bits to quadrant ',...
        'positions, whose maximum entropy is 2 bits/symbol. ',...
        'With the requested FEC rate of ',num2str(R_FEC,'%1.2f'),...
        ', the required entropy for FEC bits is ',...
        num2str(H_quad,'%1.2f'),', which exceeds the 2 bits/sym limit!',...
        ' Consider increasing the FEC rate. The minimum FEC rate '...
        'allowed for this system is (log2(M)-2)/log2(M)',...
        num2str(R_FEC_min,'%1.4f'),'.']);
end

% 计算概率整形的关键熵值（不用深算，记含义）
H_PAS = nBpS * R_FEC + H_quad;  % 总目标熵
H_DM = H_PAS - 2;               % 分布匹配的熵
H_PS = nBpS * R_FEC;            % 概率整形的熵

%核心逻辑：概率整形需要给 "四象限选择" 留 2 比特的熵空间（因为 QAM 星座图分 4 个象限），如果 FEC 编码占用的熵超过 2 比特，就会冲突，所以代码会报错提醒；
%flag的作用：如果原始星座点数不是 2 的整数次幂（比如不是 4/16/64），就自动扩展成标准 QAM 星座（保证后续算法能运行）。


%% Apply Distribution Matcher  核心操作 —— 应用分布匹配实现概率整形（代码的 "核心动作"）
[Stx,txSyms,QAM.R_CCDM] = Tx_PS_CCDM(C,H_PAS,txBits);
%Tx_PS_CCDM：是实现 "CCDM 分布匹配" 的函数（不用纠结细节，记功能即可）；
% 功能：根据计算出的目标熵H_PAS，把输入的txBits（等概率比特）转换成 "非等概率的符号索引txSyms"—— 具体来说，让低功率星座点的索引出现概率更高，高功率的更低；
% 输出：
% Stx：初步的整形后复信号；
% txSyms：初步的整形后符号索引；
% QAM.R_CCDM：CCDM 算法的实际码率（保存到 QAM 结构体里）；


%%
if flag    % 如果之前扩展了星座图（flag=true）
    QAM.M = M_PS;  % 更新QAM阶数为扩展后的数
    % 重新生成标准QAM星座图（格雷编码）
    QAM.IQmap = qammod(0:M_PS-1,M_PS,'gray').';    % 把整形后的信号重新映射到新星座图，得到新的符号索引
    txSyms = signal2symbol(Stx,QAM.IQmap);
    nBpS = log2(M_PS); % 更新每符号比特数

    % 生成"符号→比特"的映射表（方便后续解调
    bMap = false(M_PS,nBpS);
    for n = 0:M_PS-1
        [~,e] = log2(n);
        bMap(n+1,:) = rem(floor(n * pow2(1-max(e,nBpS):0)),2);
    end
    QAM.sym2bitMap = bMap;  % 保存映射表
    QAM.nBpS = log2(M_PS);   % 保存每符号比特数
end
% 如果之前扩展了星座图，就重新计算所有相关参数，保证QAM结构体里的信息和新星座图一致。
% 核心动作：qammod(0:M_PS-1,M_PS,'gray').' → 生成标准的格雷编码 QAM 星座图（格雷编码的好处是 "相邻星座点只有 1 个比特不同"，抗误码）；
% signal2symbol：把复信号重新映射回新星座图的符号索引（相当于 "根据地址找包裹编号"）。


%% Set QAM Parameters  计算并保存整形后的关键参数（记录 "整形效果"）
QAM.maxConstPower = max(abs(Stx).^2);   % 星座图最大功率
QAM.meanConstPower = mean(abs(Stx).^2);  % 星座图平均功率（整形后会降低）
QAM.maxConstPower = max(abs(Stx).^2);   % 重复计算（笔误，不影响）

% 计算每个星座点的使用概率
edges = -0.5:1:QAM.M-0.5;  % 直方图的边界
QAM.symProb = histcounts(txSyms(:),edges,'Normalization','prob').';

% 计算星座图的熵（衡量整形效果：熵越低，整形越明显）
tmp = log2(QAM.symProb);
tmp(isinf(tmp)) = 0;  % 处理0概率的情况（log2(0)是无穷大，设为0）
QAM.entropy = -sum(QAM.symProb.*tmp);% 熵的计算公式
% 核心目的：把概率整形后的关键指标保存到QAM结构体里，方便后续分析 / 解调使用；
% 关键指标：
% meanConstPower：平均功率 —— 概率整形的核心收益就是降低平均功率（因为多用低功率点）；
% symProb：每个星座点的使用概率 —— 能看到 "低功率点概率高、高功率点概率低"，这就是整形的效果；
% entropy：熵值 —— 熵越低，说明星座点使用概率的不均匀性越强，整形效果越好；




%% Debug
% RGB = fancyColors;
% plotProbShaping_PDF_const('const',QAM.IQmap,...
%     'symbols',txSyms(:),'color',RGB.itred);
% plotConstMap(QAM.IQmap,QAM.sym2bitMap,QAM.symProb);

