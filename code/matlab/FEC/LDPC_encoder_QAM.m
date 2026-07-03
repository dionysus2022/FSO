%"LDPC 编码 + QAM 调制" 
% 这段代码是通信发射端的核心编码 + 调制流程，核心作用是：
% 对原始比特流（txBits）做 LDPC 编码（前向纠错编码，提升抗干扰能力）；
% 把编码后的比特流转换成 QAM 符号（比特→符号映射）；
% 把 QAM 符号转换成最终的发射信号（符号→复数值信号）；
% 全程保证比特数是 "LDPC 码块长度" 和 "每符号比特数" 的整数倍，避免数据错位。
function [Stx,txSyms,txBits,FEC] = LDPC_encoder_QAM(txBits,FEC,C)

% Last Update: 09/11/2019


%% Input Parameters
[nPol,nBits] = size(txBits);  % 比特流维度：偏振数×总比特数
nBpS = log2(numel(C));  % 每符号比特数（星座点数→比特数，比如16QAM→log2(16)=4）
R_FEC = FEC.rate;   % LDPC编码率（比如0.8→100个信息比特→125个编码比特）

%% Initialize LDPC Encoder 初始化LDPC编码器
parityCheckMatrix = dvbs2ldpc(R_FEC);  % 生成DVB-S2标准的LDPC校验矩阵（成熟的行业标准）
LDPC_enc = comm.LDPCEncoder(parityCheckMatrix);  % 创建LDPC编码器对象
[nRows,nCols] = size(LDPC_enc.ParityCheckMatrix);  % 校验矩阵维度：行数=校验比特数，列数=编码后块长
LDPC_blockLength = nCols - nRows;  % 信息块长度=编码后块长-校验比特数（比如125-25=100）
LDPC_nBlocks = nBits/LDPC_blockLength;   % 原始比特流包含的LDPC码块数

%% Truncate txBits to Guarantee Integer Number of Symbols  截断比特数到合法长度（避免错位，核心细节）
flag = true;
while flag
    LDPC_nBlocks = nBits/LDPC_blockLength;  % 计算当前比特数对应的LDPC码块数
    nSyms = nBits/nBpS;                      % 计算当前比特数对应的符号数
      % 检查：LDPC码块数和符号数都是整数（误差<1e-6视为整数）
    if LDPC_nBlocks - floor(LDPC_nBlocks) < 1e-6 && ... 
        nSyms - floor(nSyms) < 1e-6
        flag = false;   % 满足条件，停止截断
    else
        nBits = nBits - 1;   % 不满足，删掉最后1个比特
    end
end
txBits = txBits(:,1:nBits);  % 截断比特流到合法长度

%% Apply LDPC Encoder  执行LDPC编码
idx_FEC = [];  % 初始化FEC索引（标记校验比特的位置）
for k = 1:LDPC_nBlocks     % 逐LDPC码块编码
    % 第k个码块的原始比特索引
    idx_block = (k-1)*LDPC_blockLength+1:k*LDPC_blockLength;
    % 第k个码块编码后的比特索引
    idx_enc = (k-1)*nCols+1:k*nCols;
    for n = 1:nPol   % 逐偏振编码（双偏振则处理2路）
        % 对当前偏振的当前码块比特做LDPC编码（转置是适配Matlab编码器输入格式）
        encBits(n,idx_enc) = LDPC_enc(txBits(n,idx_block)')';
    end
    % 记录当前码块的校验比特索引（编码后块长中，信息比特后是校验比特）
    idx_FEC = [idx_FEC idx_enc(LDPC_blockLength+1:end)];
end

%% Convert Bits to Symbols
txSyms = bit2sym(encBits,nBpS);

%% Convert Symbols to Transmitted Signal
Stx = symbol2signal(txSyms,C);

%% Output Parameters
FEC.LDPC_enc = LDPC_enc;
FEC.idx = idx_FEC;
