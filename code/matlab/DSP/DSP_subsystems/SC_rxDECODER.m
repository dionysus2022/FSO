clear;clc
function [DECODER] = SC_rxDECODER(DEMAPPER,BIT,FEC,PCS)
%DEMAPPER - 解调器输出的数据（包含软信息LLRs等）
% BIT - 发送端的比特信息（用于对比和同步）
% FEC - 前向纠错码的参数（如LDPC校验矩阵）
% PCS - （可选）概率整形相关参数


% Last Update: 09/11/2019


%% Entrange Message
entranceMsg('BIT DECODER');
tic

%% Input Parser
% nargin 是MATLAB内置变量，表示函数输入参数的个数
% 根据是否有PCS参数判断是否使用概率整形技术

if nargin == 4
    isPCS = true; %如果输入了4个参数，说明使用了概率整形
else
    isPCS = false; %否则就是普通模式
end

%% Input Parameters  提取输入参数
PCM_FEC = FEC.LDPC_enc.ParityCheckMatrix; %LDPC校验矩阵
LDPC_blockLength = size(PCM_FEC,2);
nIter_FEC = FEC.nIter;
idx_FEC = FEC.idx;
LLRs = DEMAPPER.LLRs.';%对数似然比（软信息），转置成行向量
% LLRs(对数似然比)：接收端对每个比特的"可信度"度量 
% 正数——可能是比特1；负数——可能是比特0；绝对值越大——可信度越高
if isPCS
    txBits_afterFEC = BIT.txBits_afterPAS;
else
    txBits_afterFEC = BIT.txBits_afterFEC;
end

%% Rx Decoder （接收端解码模式）
if isPCS  %使用概率整形
    txBits = DEMAPPER.txBits;
    % Synchronize Bits:  步骤1：比特同步
    [txBits,LLRs,idx_FEC,SYNC] = FEC_syncBits(txBits,...
        LLRs,idx_FEC,txBits_afterFEC,LDPC_blockLength);
    % Apply FEC Decoder: 步骤2：LDPC解码
    rxBits_afterFEC = LDPC_decoder(LLRs.',PCM_FEC,idx_FEC,nIter_FEC);
    nBits_afterFEC = length(rxBits_afterFEC);
    % Apply Inverse Distribution Matcher:  步骤3：逆分布匹配（恢复原始比特）
    DECODER = PAS_bitReceiver(BIT.txBits,rxBits_afterFEC,PCS);
    
    txBits_afterFEC = txBits(:,setdiff(1:length(txBits),idx_FEC));
    DECODER.txBits_afterFEC = txBits_afterFEC(:,1:nBits_afterFEC);
    DECODER.rxBits_afterFEC = rxBits_afterFEC;
    DECODER.SYNC = SYNC;
   

else %普通模式（无概率整形）
    rxBits_afterFEC = LDPC_decoder(LLRs.',PCM_FEC,idx_FEC,nIter_FEC);
    %直接LDPC解码，没有同步于泥整形步骤
    nBits_afterFEC = length(rxBits_afterFEC);
    DECODER.txBits_afterFEC = txBits_afterFEC(:,1:nBits_afterFEC);
    DECODER.rxBits_afterFEC = rxBits_afterFEC;
end

%% Elapsed Time
elapsedTime = toc;
myMessages(['Bit Decoder - Elapsed Time: ',...
    num2str(elapsedTime,'%1.4f'),' [s]\n'],1);
end
 %解码后的数据存入DECODER结构体
% txBits_afterFEC：发送端经过FEC编码后的比特（参考）
% rxBits_afterFEC：接收端解码恢复的比特
% SYNC：同步相关信息（仅PCS模式）