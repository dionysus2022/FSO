function [STX_ofdm] = OFDM_ps(STX,ofdm,sizeofqamsymbol)

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

SignalBaseI    = real(ifft(Spectral').*sqrt(NumberOfIFFTSamples));
SignalBaseQ    = imag(ifft(Spectral'));

% Cyclic prefix
A = SignalBaseI( NumberOfIFFTSamples - NumberOfGuardTime + 1 : NumberOfIFFTSamples , : );
B = SignalBaseQ( NumberOfIFFTSamples - NumberOfGuardTime + 1 : NumberOfIFFTSamples , : );

TimeSignalPlusGuardI = [A ; SignalBaseI];%把124到128填充到序列最前面
TimeSignalPlusGuardQ = [B ; SignalBaseQ];

%%% Interpolation并转串
SignalTimeI = reshape(TimeSignalPlusGuardI, 1, size(TimeSignalPlusGuardI,1)*size(TimeSignalPlusGuardQ,2));
SignalTimeQ = reshape(TimeSignalPlusGuardQ, 1, size(TimeSignalPlusGuardQ,1)*size(TimeSignalPlusGuardQ,2));

STX_ofdm = SignalTimeI + 1i .* SignalTimeQ;

end

