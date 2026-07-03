% 这段代码的作用是：把一串二进制比特（比如 010110...），转换成通信里能传输的 QAM 调制信号。
% 输入：二进制比特流 + QAM 调制参数（比如用 16QAM 还是 4QAM、编码方式等）
% 输出：调制后的复基带信号（可以直接用于仿真 / 硬件传输）+ 中间的符号索引
function [Stx,txSyms] = Tx_QAM(QAM,txBits)
%Stx-调制后的复基带信号,txSyms - 比特映射后的符号索引

% Last Update: 30/09/2018


%% Generate Symbols from Bits
%两种编码模式的处理逻辑
switch QAM.encoding
    case 'normal' %普通编码
        txSyms = bit2sym(txBits,log2(QAM.M)); % 第一步：比特→符号索引，转成十进制的符号索引
        Stx = symbol2signal(txSyms,QAM.IQmap); % 第二步：符号索引→复基带信号
    case 'diff-quad'%查分四进制编码
        txSyms = bit2sym_DiffQuad(txBits,log2(QAM.M));
        Stx = symbol2signal(txSyms,QAM.IQmap);   
end

% 这段代码的核心是两步转换：比特→符号索引→复基带信号，实现 QAM 调制；
% switch-case是根据编码方式选不同的 "打包工具"（普通 / 差分）；
% log2(QAM.M)是计算 "每个符号装几个比特"，是 QAM 调制的基础。