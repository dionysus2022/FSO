function [Stx,txSyms,R_CCDM] = Tx_PS_CCDM(C,H,txBits)

% Last Update: 08/01/2019


%% Input Parameters
M = numel(C);
[nPol,nBits] = size(txBits);
nSyms = ceil(nBits/log2(M));%786432/6=131072=2^17

%% Assign Symbol Probability According to Maxwell-Boltzman Distribution
lambda =0.0222%entropy2lambda(H,C);%由熵得lambda
symProb = exp(-lambda*abs(C).^2);%确定好lambda后得到了MB分布的分子，分母为所有加和，共有64个，C是星座点数组即每个幅值都有对应概率，即公式中的距离原点距离x，其中.^意为64个每个元素分别做平方

%% Initialize CCDM
[symProb,nBitsInfo,symFreq] = ccdm.initialize(symProb,nSyms);%innitialze的目的是确定64每个输出幅度在n中出现了几次，这里称为freq频次，即装好bag。以及确定好输入位数nBitsInfo
R_CCDM = nBitsInfo/nBits;%567723/786432

%% Encode with Distribution Matcher
[Stx,txSyms] = deal(NaN(nPol,nSyms));
for n = 1:nPol
    i_TX = ccdm.encode(txBits(n,1:nBitsInfo),symFreq).' + 1;%理论输入位数是786432，实际上只有567723
    Stx(n,:) = C(i_TX).';%真正要传出去的幅值
    txSyms(n,:) = i_TX.'-1;%i_TX和txsyms只是一个索引，索引C中64个幅值中的哪一个
end

