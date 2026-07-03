function [tx_signal,Spectral] = OFDM(STX,ofdm,sizeofqamsymbol)
Ltrs_num      = 2;
head_zeros    = 80;    % 帧头0的个数
IQXmatrix = STX;
% OFDM Tx parameters
NumberOfIFFTSamples   = ofdm.NumberOfIFFTSamples;
Carrier_location      =ofdm.Carrier_location;
NumberOfCarriers      = length(Carrier_location);
NumberOfGuardTime     = ofdm.NumberOfGuardTime;

zeroPadFactor       = NumberOfIFFTSamples/NumberOfCarriers;
% Number of M_ary symbols
NrOfDigitalSymbols  = sizeofqamsymbol; 
% Number of OFDM symbols that can be entirely transmitted in the time window
%NrOfOFDMSymbols     = floor(NrOfDigitalSymbols*zeroPadFactor/(NumberOfIFFTSamples+NumberOfGuardTime)); %1024*1.6=1638意为要完全表示1024个信息，如果是一排的话，需要1638个载波，也就是需要FFT之后有1638个时点。而FFT之后每个OFDM符号有133个时点（加了5个点的循环前缀），所以1638个时点每一排都是133个时点的话，1638/133=12.3多出来的0.3无法组成另一个ofdm符号，故只能取12个ofdm符号，而且只能取1024前对应960个
NrOfOFDMSymbols  =  ofdm.size;

%IQXmatrix = reshape(IQX(1:NrOfOFDMSymbols*NumberOfCarriers),NrOfOFDMSymbols,NumberOfCarriers );%只取1024的前960个
  
% IFFT
% Zero-padding for IFFT
Spectral            = zeros( NrOfOFDMSymbols , NumberOfIFFTSamples);%因为128里除了80以外的要填0所以尺寸是128

for i = 1:NrOfOFDMSymbols
    Spectral(i,Carrier_location)     = IQXmatrix(i,:);%只在有location的位置填上数据
    Spectral(i,NumberOfIFFTSamples/2+2:NumberOfIFFTSamples) =conj(Spectral(i,NumberOfIFFTSamples/2:-1:2));
end

SignalBaseI    = real(ifft(Spectral.').*sqrt(NumberOfIFFTSamples));%乘sqrt(256)是为了让时域频域能量保持一致
SignalBaseQ    = imag(ifft(Spectral.'));

% Cyclic prefix
A = SignalBaseI( NumberOfIFFTSamples - NumberOfGuardTime + 1 : NumberOfIFFTSamples , : );
B = SignalBaseQ( NumberOfIFFTSamples - NumberOfGuardTime + 1 : NumberOfIFFTSamples , : );

TimeSignalPlusGuardI = [A ; SignalBaseI];%把124到128填充到序列最前面
TimeSignalPlusGuardQ = [B ; SignalBaseQ];

%%% Interpolation并转串
SignalTimeI = reshape(TimeSignalPlusGuardI, 1, size(TimeSignalPlusGuardI,1)*size(TimeSignalPlusGuardI,2));
SignalTimeQ = reshape(TimeSignalPlusGuardQ, 1, size(TimeSignalPlusGuardQ,1)*size(TimeSignalPlusGuardQ,2));

STX_ofdm = SignalTimeI + 1i .* SignalTimeQ;

%% 生成长训练序列
load 'LongTrainSym_ini.mat'                                    % LongTrainSym_ini为1×1024矩阵，第一个元素是0，后面是1或-1                                                        
LongTrainSym = LongTrainSym_ini(1:NumberOfIFFTSamples);                     % LongTrainSym = LongTrainSym_ini(1:256)
LongTrainSym([1 129])=0;                                   % LongTrainSym([1 129])=0
ltrs_ifft_in = LongTrainSym;                                   
ltrs_ifft_in(1,NumberOfIFFTSamples/2+2:NumberOfIFFTSamples) =conj(ltrs_ifft_in(1,NumberOfIFFTSamples/2:-1:2));
ltrs_ifft_out = ifft(ltrs_ifft_in); % ifft_lin和ifft_lout为1×256实数矩阵
if Ltrs_num==1   %训练序列个数
	long_trs = [ltrs_ifft_out(length(ltrs_ifft_out) - 1 * NumberOfGuardTime + 1: length(ltrs_ifft_out)), ltrs_ifft_out]; % 加CP（将后CPLength列加到前面）long_trs为1×80实数矩阵,long_trs = [ifft_lout(64-15:64), ifft_lout]
elseif Ltrs_num==2
	long_trs = [ltrs_ifft_out(length(ltrs_ifft_out) - 1 * NumberOfGuardTime + 1: length(ltrs_ifft_out)), ltrs_ifft_out,ltrs_ifft_out];
else
	error('请输入正确的长训练序列个数！1或2');
end

%% 并串转换，添加长训练序列和80个空子载波（前导），生成帧
tx_signal = [];
preamble = [zeros(1,head_zeros), long_trs]; % 加80个空子载波，preamble为1×160实数矩阵
    tx_signal = [tx_signal, preamble, STX_ofdm]; %加上前导，为1×8160实数矩阵

end

