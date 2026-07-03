%前向纠错帧同步
%核心作用是FEC 帧同步：用已知参考序列找同步点，截断错位比特，重新整理 FEC 索引；
function [txBits_demap,LLRs,idx_FEC,SYNC] = FEC_syncBits(txBits_demap,...
    LLRs,idx_FEC,txBits_afterPAS,LDPC_blockLength)

% Last Update: 09/11/2019


%% Expand TX Bits Reference to the Length of LLRs 扩展参考序列到接受序列长度
%核心目的：让参考序列（TX）和接收序列（RX）长度完全一致，才能逐位比对找同步点；

TX = txBits_afterPAS*2-1;  % 0/1比特→±1序列（方便后续符号比对：0→-1，1→1）
nTX = length(TX);  % 参考序列长度
RX = sign(LLRs);   % 对LLR取符号→硬判决±1序列（LLR>0→1→+1，LLR<0→0→-1）
nRX = length(RX);   % 接收序列长度  
nBlocks = floor(nRX/nTX);   % 参考序列需要重复的次数（比如RX长1000，TX长100→10次）
remSamples = nRX - nBlocks*nTX;  % 剩余不足一个TX长度的比特数
% 扩展参考序列：重复nBlocks次 + 补剩余比特 → 长度和RX一致
TX = [repmat(TX,1,nBlocks) TX(:,1:remSamples)];

%% Find Sync Point  找同步点
SYNC.method = 'complexField';  % 同步方法：复域相关（不用纠结细节，是成熟的同步算法）
% SYNC.debug = true;      
SYNC.minDelay = 0;   % 最小搜索时延（从0开始）
SYNC.maxDelay = nTX;    % 最大搜索时延（不超过一个参考序列长度）
% 调用同步函数：比对TX和RX，找到最优同步点（时延）
[~,SYNC] = SC_syncTxRx(TX,RX,1,SYNC);

%% Truncate LLRs and txBits to start in the first full FEC frame  截断数据到同步点（修正帧偏移）
delay = SYNC.syncPoint - 1;  % 同步点对应的时延（从0开始计数）
% 计算时延对应的LDPC码块数和剩余比特
nBlocksDelay = floor(delay/LDPC_blockLength);   % 偏移了多少个完整LDPC块
remDelay = delay - nBlocksDelay*LDPC_blockLength;  % 剩余不足一个块的比特数
% 计算最终同步点：从第一个完整LDPC块开始
syncPoint = (nBlocksDelay+1)*LDPC_blockLength - remDelay + 1;
% 截断LLR和硬判决比特：从同步点开始，删掉前面的偏移比特
LLRs = LLRs(:,syncPoint:end);
txBits_demap = txBits_demap(:,syncPoint:end);
%同步点可能不在 "完整 FEC 帧的起始"，需要调整到第一个完整 LDPC 码块的开头，避免解码时帧错位；

%重新整理 FEC 索引（适配同步后的比特流）
%核心目的：idx_FEC原本是 "单个 FEC 块的索引"，同步后的比特流包含多个块，需要把索引扩展到所有块；
nBits_perBlock = length(txBits_afterPAS);  % 每个参考块的比特数
nBlocks = floor(length(LLRs)/nBits_perBlock); % 同步后有多少个完整参考块
nTail = mod(length(LLRs),nBits_perBlock);   % 剩余不足一个块的比特数
idx_FEC_all = [];
% 扩展索引到所有完整块：每个块的索引偏移"块数×块长度"
for n = 1:nBlocks
    idx_FEC_all = [idx_FEC_all idx_FEC + (n-1)*nBits_perBlock];
end
% 补充剩余比特的索引：只取索引≤剩余比特数的部分，偏移到最后一个块后
idx_FEC = [idx_FEC_all idx_FEC(idx_FEC<=nTail) + nBlocks*nBits_perBlock];
