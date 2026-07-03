function [SRX_deofdm] = deOFDM_ps(SRX,ofdm,sizeofqamsymbol)
%% Serial to parallel conversion
SignalTimeIX=real(SRX);
SignalTimeQX=imag(SRX);

NumberOfIFFTSamples   = ofdm.NumberOfIFFTSamples;
Carrier_location      =ofdm.Carrier_location_demo;
NumberOfCarriers      = length(ofdm.Carrier_location_demo);
NumberOfGuardTime     = ofdm.NumberOfGuardTime;

zeroPadFactor       = NumberOfIFFTSamples/NumberOfCarriers;
% Number of M_ary symbols
NrOfDigitalSymbols  = sizeofqamsymbol; 
% Number of OFDM symbols that can be entirely transmitted in the time window
%NrOfOFDMSymbols     = floor(NrOfDigitalSymbols*zeroPadFactor/(NumberOfIFFTSamples+NumberOfGuardTime)); %1024*1.6=1638意为要完全表示1024个信息，如果是一排的话，需要1638个载波，也就是需要FFT之后有1638个时点。而FFT之后每个OFDM符号有133个时点（加了5个点的循环前缀），所以1638个时点每一排都是133个时点的话，1638/133=12.3多出来的0.3无法组成另一个ofdm符号，故只能取12个ofdm符号，而且只能取1024前对应960个

NrOfOFDMSymbols = ofdm.size;

SignalBaseIX = reshape(SignalTimeIX, NumberOfIFFTSamples + NumberOfGuardTime, NrOfOFDMSymbols);
SignalBaseQX = reshape(SignalTimeQX, NumberOfIFFTSamples + NumberOfGuardTime, NrOfOFDMSymbols);

%% Remove cyclic prefix

SignalBaseIX = SignalBaseIX(NumberOfGuardTime + 1 : NumberOfIFFTSamples + NumberOfGuardTime, :);
SignalBaseQX = SignalBaseQX(NumberOfGuardTime + 1 : NumberOfIFFTSamples + NumberOfGuardTime, :);

%% FFT
SpectralIX    = fft(SignalBaseIX)./sqrt(NumberOfIFFTSamples);
SpectralQX    = fft(SignalBaseQX)./sqrt(NumberOfIFFTSamples);

SpectralIQX = SpectralIX + 1i.*SpectralQX;

%% Carrier location

SpectralrcX   = SpectralIQX';

%%% did fftshift before ifft, need to do fftshift again in the receiver
IQXmatrix = zeros(NrOfOFDMSymbols,NumberOfCarriers);

for i = 1:NrOfOFDMSymbols
    IQXmatrix(i , :)                     = SpectralrcX(i , Carrier_location);
end
%% Normalization

SRX_deofdm = reshape(IQXmatrix,1,size(IQXmatrix,1)*size(IQXmatrix,2));
end

