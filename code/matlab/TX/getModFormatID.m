%作用：根据输入的QAM 阶数 M（比如 4、16、32）和编码方式 encoding（普通 / 差分），自动判断 QAM 的类型（正方形 / 十字形），并生成标准化的 "调制格式名称 / ID"
% 输入：M（QAM 阶数，比如 4、16、32）、encoding（编码方式，可选，默认normal）；
% 输出：
% MF_ID：调制格式唯一 ID（比如16QAM_square、QPSK_DiffQuad）；
% modFormat：简化的调制格式名（比如16QAM、QPSK）；
% mode_QAM：QAM 星座图类型（square-QAM正方形 / cross-QAM十字形）
function [MF_ID,modFormat,mode_QAM] = getModFormatID(M,encoding)

% Last Update: 16/05/2018


%% Input Parser
if nargin == 1    % 如果只传入了M，没传encoding
    encoding = 'normal'; % 编码方式默认设为'normal'（普通编码）
end

%% Set Modulation Format ID  判断 QAM 类型并生成格式 ID
%第一类：M是平方数（比如4=2²、16=4²、81=9²）且M≥4 → 正方形QAM
if mod(sqrt(M),1) == 0 && M >=4
    mode_QAM = 'square-QAM';   % 标记星座图类型：正方形
    if M > 4  % 比如M=16、64
        modFormat = [num2str(M),'QAM'];  % 生成格式名：16QAM、64QAM
        MF_ID = [modFormat, '_square'];   % 生成唯一ID：16QAM_square、64QAM_square
    elseif M == 4   % 特殊：4QAM就是QPSK
        modFormat = 'QPSK';  % 4QAM的标准名称是QPSK（四相移键控）
        MF_ID = modFormat;  % ID就是QPSK
    end

 % 第二类：M不是平方数，但M是2的整数次幂（比如8=2³、32=2⁵、128=2⁷）   
elseif mod(log2(M),1) == 0
    if M > 2  % 比如M=8、32
        mode_QAM = 'cross-QAM';  % 标记星座图类型：十字形
        modFormat = [num2str(M),'QAM'];   % 格式名：8QAM、32QAM
        MF_ID = [modFormat, '_cross'];   % 唯一ID：8QAM_cross、32QAM_cross
    elseif M == 2  % 特殊：2QAM就是BPSK
        mode_QAM = 'square-QAM';    % 2QAM（BPSK）的星座图是正方形（只有两个点）
        modFormat = 'BPSK';  % 2QAM的标准名称是BPSK（二进制相移键控）
        MF_ID = modFormat;   % ID就是BPSK
    end

 % 第三类：M既不是平方数，也不是2的整数次幂（比如6、10、12）→ 报错
else
    error('The QAM_config function is only compatible with QAM constellation of size 2^n or n^2, for any integer n. The parsed constellation size, %d, does not fulfill this condition. Please consider changing the constellation size.',M);
end

% 补充：如果编码方式是差分四进制（diff-quad）且M≥4 → ID加后缀_DiffQuad
if strcmp(encoding,'diff-quad') && M >= 4
    MF_ID = [MF_ID '_DiffQuad'];
end
