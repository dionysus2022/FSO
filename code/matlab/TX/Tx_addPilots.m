%作用：插入已知导频符号：在原始信号的指定位置插入功率匹配的 QPSK 导频，供接收端做信道估计；
%根据指定的导频插入率和导频类型，在原始发射信号Stx的指定位置插入已知的导频符号，输出包含导频的信号Stx，并记录导频的位置 / 序列，供接收端使用。
function [Stx,PILOTS] = Tx_addPilots(Stx,PILOTS,C)

% Last Update: 31/03/2019


%% Input Parser 导频类型默认值补全（基础准备）
if isfield(PILOTS,'option')
    pilotOption = PILOTS.option; % 提取导频类型（比如'meanQPSK'）
else
    pilotOption = 'meanQPSK';  % 默认导频类型：功率匹配的QPSK
end

%% Input Parameters  提取基础参数（核心变量初始化）
[nPol,nSyms] = size(Stx); % 原始信号维度：nPol=偏振数，nSyms=原始符号数
pilotRate = PILOTS.rate;  % 导频插入率（比如1/10 → 每10个符号插1个导频）
meanP = mean(abs(Stx(:)).^2);%原始信号的平均power（用于匹配导频功率）
%关键：mean(abs(Stx(:)).^2) 是计算复信号的平均功率 —— 复信号的功率 = 幅度的平方（abs(x)^2），再求均值。


%% Calculate Symbol Indices for Tx Pilots and Payload  计算导频 / 有效载荷的位置（核心逻辑 1：插在哪？）
% 步骤1：把导频率（小数）转成分数（比如0.1→1/10，0.2→1/5）
[A,B] = rat(pilotRate);  % A=分子，B=分母 → 导频率=A/B
% 步骤2：确定每B个符号中，哪些位置插导频（比如A=1,B=10 → 位置2:10？不，看例子）
idx_pilots = A+1:B;% 导频位置的"基础偏移"（比如A=1,B=10 → 2:10？实际是"每B个符号插A个"）
% 步骤3：计算能完整插入的导频块数（比如原始符号数1000，B=10 → 100块）
nAB = floor(nSyms/A); % 向下取整，保证导频块数为整数
% 步骤4：基础导频位置的数量
nPilots = numel(idx_pilots);
% 步骤5：生成所有导频的位置（重复基础位置+块偏移）
idx_pilots = repmat(idx_pilots,1,nAB) + ...
    B*(rectpulse(1:nAB,nPilots) - 1);
% 步骤6：更新导频总数、插入导频后的总符号数
nPilots = numel(idx_pilots);
nSyms_withPilots = nSyms + nPilots;
% 步骤7：有效载荷（原始信号）的位置=总位置 - 导频位置
idx_payload = setdiff(1:nSyms_withPilots,idx_pilots);
% 步骤8：扩展位置到所有偏振（双偏振则每个偏振的导频位置相同）
idx_pilots = repmat(idx_pilots,nPol,1);
idx_payload = repmat(idx_payload,nPol,1);

%% Generate Pilot Symbols  生成导频符号
switch pilotOption
    case 'outerQPSK'   % 外层QPSK导频（星座图最外圈的QPSK点，功率大）
        C_pilot = C(abs(C) == max(abs(C)));  % 取星座图中幅度最大的点（外圈）
    case 'innerQPSK'    % 内层QPSK导频（星座图最内圈的QPSK点，功率小）
        C_pilot = C(abs(C) == min(abs(C)));  % 取星座图中幅度最小的点（内圈）
    case 'meanQPSK'   % 功率匹配的QPSK导频（工程首选）
        C_pilot = C(abs(C) == max(abs(C)));   % 先取外圈QPSK点
        % 缩放导频功率，使其等于原始信号的平均功率
        C_pilot = C_pilot * sqrt(meanP) / ...
            sqrt(mean(abs(C_pilot).^2));
    case 'customQPSK'    % 自定义缩放的QPSK导频
        C_pilot = C(abs(C) == max(abs(C)));
         % 功率匹配+自定义缩放因子
        C_pilot = C_pilot * sqrt(meanP) / ...
            sqrt(mean(abs(C_pilot).^2)) * PILOTS.scaleFactor;
end
C_pilot = C_pilot.';  % 转置（适配信号维度）
% 随机选择导频符号（从C_pilot中随机选，保证导频序列的随机性）
Stx_pilot = C_pilot(randi(numel(C_pilot),[nPol nPilots]));

%% Add Pilot Symbols to the Transmitted Signal  插入到频道信号中
% 初始化包含导频的信号矩阵（NaN表示未赋值）
Stx_pilots = NaN(nPol,nSyms_withPilots);
for n = 1:nPol  % 逐偏振处理
    Stx_pilots(n,idx_pilots(n,:)) = Stx_pilot(n,:);   % 导频位置赋值导频符号
    Stx_pilots(n,idx_payload(n,:)) = Stx(n,:);   % 有效载荷位置赋值原始信号
end
Stx = Stx_pilots;   % 更新为包含导频的信号

%% Assign PILOTS Parameters  保存导频参数
PILOTS.pilotSequence = Stx_pilot;  % 保存导频序列（接收端需要知道）
PILOTS.idx = idx_pilots;  % 保存导频位置（接收端需要知道在哪找导频）

