%按需生成发射端（Tx）的二进制比特流
% 输入：nSyms（调制符号数）、M（QAM 调制阶数，如 16/64QAM）、nPol（偏振态数量，如 1/2）、BIT（参数结构体，包含比特源类型、种子等）。
% 输出：txBits（逻辑型的比特矩阵，行数 = 偏振态数，列数 = 总比特数）。
function [txBits] = Tx_generateBits(nSyms,M,nPol,BIT)

% Last Update: 03/10/2018


%% Set Default Bit Source  设置默认比特源
% 核心作用：处理 "参数缺失" 的情况，保证函数不会因为参数不全报错。
% nargin < 4：判断输入参数的个数是否少于 4 个（函数定义是 4 个输入：nSyms,M,nPol,BIT），如果是，说明用户没传BIT结构体。
% ~isfield(BIT,'source')：如果传了BIT但没指定source（比特源类型），也触发默认值。
% 最终默认：把比特源设为 randi（即 MATLAB 内置随机数生成）。
if nargin < 4 || ~isfield(BIT,'source')
    BIT.source = 'randi';
end


%% Generate Tx Bits 计算总比特数+初始化比特矩阵
nBits = floor(nSyms*log2(M));  %计算需要生成的总比特数  floor（）向下取整
txBits = NaN(nPol,nBits);%创建一个空矩阵，行数为偏振态数，列数为总比特数，初始值设置为NaN(表示“未赋值”)，后续会填充0/1
%switch-case实现“多模式选择”，三种模式 randi/PRBS/QAM-PRBS
switch BIT.source
    case 'randi' %默认。随机数生成
        if isfield(BIT,'seed')
            rng(BIT.seed); % 固定随机种子，保证结果可重复
        end
        for n = 1:nPol
            txBits(n,:) = randi(2,1,nBits)-1;  % 生成0/1比特流
        end
% 核心作用：用 MATLAB 内置的randi生成随机 0/1 比特，按偏振态分配。
% rng(BIT.seed)：如果指定了种子，就固定随机数生成器的种子（仿真中必须！否则每次运行结果不一样，无法复现）。
% randi(2,1,nBits)：生成 1 行、nBits 列的随机整数，取值为 1 或 2；
% -1：把 1→0，2→1，最终得到 0/1 的二进制比特流
%for n = 1:nPol：给每个偏振态生成独立的比特流（如双偏振则生成 2 行）。

    case 'PRBS' %基础PRBS生成
        for n = 1:nPol
            % prbs = PRBS_generator(1,nextpow2(nBits),BIT.seed+n-1);
            % txBits(n,:) = prbs(1:nBits);


            % 修正PRBS_generator调用参数：
            % num_PRBS=1（生成1个序列），deg=9（9阶，周期511≥400），num_pol=1（起始多项式1，避免越界）
            [prbs, err_flag] = PRBS_generator(1,9,1); 
            txBits(n,:) = prbs(1,1:nBits);  % 截断为需要的400长度
        end
% 核心作用：调用自定义的PRBS_generator函数生成伪随机序列，每个偏振态用不同种子（避免序列重复）。
% nextpow2(nBits)：计算大于等于 nBits 的最小 2 的幂次（PRBS 的阶数，保证 PRBS 序列长度足够）；
% 示例：nBits=4000→nextpow2 (4000)=12（因为 2^12=4096≥4000）。
% BIT.seed+n-1：给第 n 个偏振态分配不同的种子（如 seed=100，偏振态 1 用 100，偏振态 2 用 101），避免多路序列完全一样。
% prbs(1:nBits)：截断 PRBS 序列到需要的 nBits 长度（因为 PRBS 长度是 2^ 阶数 - 1，可能比 nBits 长）。
    
    case 'PRBS-QAM' %QAM专用PRBD，关联上一函数
        %txBits = QAM_PRBSgenerator(BIT,M,nPol,nSyms); %修改在下面

        [txBits,BIT] = QAM_PRBSgenerator(BIT,M,nPol,nSyms);  % 注意：原函数有两个输出，需匹配


end
% 核心作用：直接调用你上一次问的 QAM_PRBSgenerator 函数，生成适配 QAM 调制的 PRBS 比特流。
% 这是 "高级版" PRBS 生成：相比基础 PRBS，它会做串并转换、长度适配、可选延时等更贴合 QAM 系统的处理。
% 注意：这里直接把QAM_PRBSgenerator的输出赋值给txBits，替代了之前的初始化矩阵。

txBits = logical(txBits);
end
%转换为逻辑型（优化存储+适配后续处理）
% 核心作用：把数值型的 0/1 矩阵（double 类型）转换为逻辑型（logical）。
% 逻辑型在 MATLAB 中只占 1 位存储，比 double（8 字节）节省内存；
% 后续的调制、编码等操作通常要求输入逻辑型比特流，避免数值误差。
