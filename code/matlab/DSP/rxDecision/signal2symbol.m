function syms = signal2symbol(sig,C,normPower,useGPU)
%signal2symbol  Convert a complex signal into constellation symbols
%
%   This function converts an input complex signal into the corresponding
%   constellation symbols, employing minimum distance detection. 
%
%   INPUTS:
%   sig         :=  input complex signal at 1 sample/symbol [nPol x nSyms]
%   C           :=  reference constellation [M x nPol]
%                   Note that the C vector must be ordered according to the
%                   mapping between IQ contellation samples and symbol
%                   indices, i.e., C(1) corresponds to symbol #0, C(2) to
%                   symbol #1, C(3) to symbol #2, ... C(M) to symbol #M-1
%   normPower   :=  factor for normalizing the power of the reference 
%                   constellation, C [1 x nPol]
%   useGPU      :=  flag to decide whether to use or not GPU-based processing
%
%   OUTPUTS:
%   syms    :=  array of contellation symbols [nPol x nSyms]
%               The symbols are in the range 0 ... M-1, where M is the
%               constellation size
%   
%
%   Author: Fernando Guiomar
%   Last Update: 04/06/2019

%% Input Parameters
[nPol,nSyms] = size(sig);
if size(C,2) == 1%第二个维度的size值就是1
    C = repmat(C,1,nPol);%以维度1xnpol的size幅值C，这里等于没操作
end
if nargin < 4
    useGPU = false;
end

%% Normalize Signal
if nargin == 3 && ~isempty(normPower)
    if numel(normPower) == 1
        normPower = repmat(normPower,1,nPol);
    end
    for n = 1:nPol
        C(:,n) = C(:,n)*sqrt(normPower(n))/sqrt(mean(abs(C(:,n)).^2));
    end
end
%没用到
%% Symbol Decoder
syms = zeros(nPol,nSyms)-1;
for n = 1:nPol
    thisC = single(C(:,n));
    thisSig = single(sig(n,:));
    if useGPU
        thisSig = gpuArray(thisSig);
        thisC = gpuArray(thisC);
    end
    err = abs(thisSig - thisC);%是每个sig和64个C星座点分别相减，所以每个sig都有64x1大的数组,即得到每个接受向量和所有星座点的距离
%     err = abs(bsxfun(@minus,thisSig,thisC));
    [~,ind] = min(err);%~符号只作为一个占位符，相当于隐藏了第一个参数，只返回最小值索引ind。这一步是在每列，即每个sig的64x1数组里找到了最小的那个,也即完成了判决。仍组成1x131072
    if useGPU
        ind = gather(ind);
    end
    syms(n,:) = ind - 1;%这里应该是因为星座图索引是由c++弄的，数据索引是从0开始的。这里接收星座图也要和发送的相配合
end
