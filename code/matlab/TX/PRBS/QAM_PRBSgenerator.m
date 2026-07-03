%代码核心：生成适配QAM调制和偏振复用的PRBS比特流（长度匹配、串并转换、时延模拟）

function [bits,BIT] = QAM_PRBSgenerator(BIT,M,nPol,nSyms)
%输入：BIT(结构体，包含PRBS生成的参数，如种子、阶数、延时)
% M（QAM 调制阶数，如 16QAM、64QAM）、
% nPol（偏振态数量，通信领域常用，如单偏振 / 双偏振）、
% nSyms（符号数，即需要生成的调制符号总数）。
% 输出：bits（最终生成的比特流矩阵）、BIT（补全默认参数后的结构体）。
% Last Update: 30/09/2018


%% Input Parser  输入参数检查与默认值补全
if ~isfield(BIT,'seed') %判断BIT结构体里有没有seed（随机种子）这个参数。如果没有，直接报错，因为 PRBS 生成必须指定种子（保证序列可重复）
    error('You must specify the first polynomial seed for PRBS generation: PRBS.seed');
end
if ~isfield(BIT,'applyBitDelay') %是否给比特流加延时，默认设为false（不加）
    BIT.applyBitDelay = false;
end
if ~isfield(BIT,'evenLength') %是否让序列长度为偶数，默认设为true（是）
    BIT.evenLength = true;
end



%% Input  输入参数赋值
%关键解释
% PRBS阶数（degreePRBS）：PRBS 是伪随机序列，阶数决定其周期长度，比如阶数 7 的 PRBS，周期是2^7-1=127个比特（序列会重复但看起来随机）。
% N = nPol*log2(M)：QAM 调制中，每个符号需要log2(M)个比特（如 16QAM 需要 4 比特 / 符号），再乘以偏振态数量nPol，得到需要并行生成的 PRBS 数量（比如双偏振 16QAM，N=2×4=8，即同时生成 8 路 PRBS）。
% isinf(degreePRBS)：如果阶数设为无穷大，就根据需要的符号数反算阶数（保证序列长度够用）。
degreePRBS = BIT.degree;                                                    % PRBS degree  PRBS阶数（决定PRBS序列的长度，长度=2^degreePRBS -1）
evenLength = BIT.evenLength;                                                % 序列是否为偶数长度的标志flag signaling if the sequence should be of even length (in that case, one 0 must be padded at the end of each PRBS)
applyBitDelay = BIT.applyBitDelay;                                          % 是否加比特延时的标志flag signaling if bit delay should be applied to the generated/loaded PRBS sequences
seed = BIT.seed;                                                            % PRBS生成的随机种子seed for the first polynomial utilized for PRBS generation
N = nPol*log2(M);                                                           % 并行PRBS的数量（关键！）number of parallel PRBSs (only if parallel bit to symbol assignment is used)
if isinf(degreePRBS)
    nBits = floor(nSyms / log2(M));
    degreePRBS = floor(log2(nBits));
end



%% Generate PRBS  生成PRBS序列
% 核心作用：生成 N 路 PRBS 序列，优先调用自定义的PRBS_generator函数，如果调用失败（比如函数不存在），就用 MATLAB 内置的randi生成随机 0/1 矩阵（保底方案）。
% try-catch：异常处理，尝试执行PRBS_generator，失败则执行catch里的代码（抛警告 + 用 randi 生成）。
% rng(seed)：固定随机数种子，确保每次运行生成的随机序列完全一样（仿真必备）。
% cell数组bit{n}：把每一路 PRBS 序列（bitMatrix 的每一行）单独存到 cell 里，方便后续处理。

try
    %bitMatrix = PRBS_generator(N,degreePRBS,seed); %调用自定义PRBS生成函数

    % 调用专业版PRBS_generator，补充err_flag输出，避免参数越界
    [bitMatrix, err_flag] = PRBS_generator(N,degreePRBS,1); % num_pol=1，避免seed作为多项式编号越界
catch
    warning('Could not generate the specified PRBS. Proceeding with randi function.');
    rng(seed);  %固定随机种子，保证可重复
    bitMatrix = randi([0 1],N,2^degreePRBS-1); %生成N行，2^degreePRBS-1列的0/1矩阵
end
for n = 1:N
    bit{n} = bitMatrix(n,:);  % 把矩阵的每一行存到cell数组bit中（cell可存不同长度数据）
end



%% Check for Even Length Sequence 调整序列为偶数长度
%核心作用：如果evenLength为 true，给每一路 PRBS 序列末尾补一个 0，确保序列长度是偶数（通信中有时要求符号数为偶数，避免丢比特）。
if evenLength
    for n = 1:N
        bit{n} = [bit{n} 0];% 给每一路PRBS末尾补一个0，让长度变成偶数
    end
end



%% Truncate PRBS Sequence  截断/扩展序列到指定长度
% 如果指定了sequenceLength，截断到该长度
if isfield(BIT,'sequenceLength') ...
        && ~isinf(BIT.sequenceLength)
    for n = 1:N
        bit{n} = bit{n}(:,1:BIT.sequenceLength);
    end
end



%% Adjust the Length of the PRBS to the Number of Simulated Bits 根据需要的符号数调整序列长度（核心：匹配仿真所需的比特数）
nBitsPRBS = length(bit{1});
% 关键解释：
% sequenceLength：如果指定了序列长度，先截断到该长度。
% nSyms（符号数）决定最终需要的比特数：总比特数 = 符号数 × 每符号比特数（log2 (M)）。如果当前 PRBS 序列长度不够，就重复序列（repmat是矩阵重复函数）+ 补剩余比特；如果太长，就截断。
if nSyms < nBitsPRBS
    %符号数少，截断序列
    for n = 1:N
        bit{n} = bit{n}(:,1:nSyms);
    end
elseif nSyms > nBitsPRBS %符号数多，重复序列+补余
    nRep = floor((nSyms)/nBitsPRBS); % 重复次数
    nTrail = mod(nSyms,nBitsPRBS); % 剩余比特数
    for n = 1:N
        bit{n} = [repmat(bit{n},1,nRep) bit{n}(:,1:nTrail)];    
    end
end


%% Parallel to Serial Bit Assignment 并行串行转换
% 核心作用：把并行生成的 N 路 PRBS 序列，按偏振态分配并转换为串行比特流（通信中发送端需要串行发送比特）。
% zeros(nPol,...) + NaN：先创建一个空矩阵（元素为 NaN），行数是偏振态数nPol，列数是总比特数。
% k:N/nPol:end：按步长赋值，比如 N=8、nPol=2，步长 = 4，把 8 路 PRBS 分配到 2 个偏振态的串行流中。
bits = zeros(nPol,length(bit{1})*N/nPol) + NaN;
for k = 1:N/nPol
    for kk = 1:nPol
        bits(kk,k:N/nPol:end) = bit{(kk-1)*N/nPol+k};
    end
end



%% Apply Fixed Bit Delay to Bit Stream 给比特流加时延
% 核心作用：如果开启applyBitDelay，给比特流加一个固定 / 随机的延时（通信仿真中模拟信号传输的时延）。
% circshift：MATLAB 的循环移位函数，[0 bitDelay]表示列方向移位（比特流是按列存储的），比如延时 3，第 1 列变第 4 列，最后 3 列补到开头。
if applyBitDelay
    if isfield(BIT,'bitDelay') && BIT.bitDelay
        bitDelay = BIT.bitDelay;
    else
        bitDelay = randi(length(bits),1,1);% 随机生成延时值
    end
    bits = circshift(bits, [0 bitDelay]); % 循环移位（比特流整体后移，末尾补到开头）
end
%执行逻辑梳理（整体流程）
%1.检查输入参数，补全默认值→2. 计算并行 PRBS 数量 N→3. 生成 N 路 PRBS 序列（失败则用 randi 保底）
% →4. 调整序列长度（补 0 / 截断 / 重复）→5. 并行转串行，分配到不同偏振态→6. 可选加比特延时→输出最终比特流。