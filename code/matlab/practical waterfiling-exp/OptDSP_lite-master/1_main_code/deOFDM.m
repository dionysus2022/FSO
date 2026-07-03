function [SRX_deofdm] = deOFDM(SRX,ofdm,sizeofqamsymbol)
Ltrs_num      = 2;
zeros_head    = 80;    % 帧头0的个数
plot_en=1;
Fg            = 1e9;  % 信号带宽
LtrsCPLength=ofdm.NumberOfGuardTime;

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
%计算帧长
symbol_bits = zeros_head + NumberOfGuardTime + NumberOfIFFTSamples * Ltrs_num + (NumberOfIFFTSamples + NumberOfGuardTime) * sizeofqamsymbol; %帧长=8160
rx_signal_recovered=SRX;
freq_data_syms_out=[];
%% 生成长训练序列
load 'LongTrainSym_ini.mat'                                    % LongTrainSym_ini为1×1024矩阵，第一个元素是0，后面是1或-1                                                        
LongTrainSym = LongTrainSym_ini(1:NumberOfIFFTSamples);                     % LongTrainSym = LongTrainSym_ini(1:256)
LongTrainSym([1 129])=0;                                   % LongTrainSym([1 129])=0
ltrs_ifft_in = LongTrainSym;                                   
ltrs_ifft_in(1,NumberOfIFFTSamples/2+2:NumberOfIFFTSamples) =conj(ltrs_ifft_in(1,NumberOfIFFTSamples/2:-1:2));
LongTrainSym =ltrs_ifft_in;

% 计算第一帧出现的位置
    [detected_packet, edge_index] = packet_edge_power_dect(rx_signal_recovered(1:2*symbol_bits),zeros_head);  % 基于延时相关算法的edge搜索，理想时输出帧头位置，帧头连0处的第一个0位置
    [fine_time_est,data_df,max_peak_long] = rx_fine_time_sync_cross_corr(detected_packet,LtrsCPLength,LongTrainSym,zeros_head,plot_en); % 输出第一个训练序列的开始位置
    fine_index(1) = edge_index + fine_time_est - 1;                     % 帧头第一个0的位置+训练序列的开始位置-1
    frame_length = floor(length(rx_signal_recovered)/symbol_bits)-1;    
    % 计算数据一共有多少帧
for loop=1%:frame_length

    fine_index_temp = fine_index(loop);
    rx_signal = rx_signal_recovered(fine_index(loop) - (zeros_head + LtrsCPLength-5) :fine_index(loop)+symbol_bits-1); % 一帧数据的大致范围，前面空载波减去了5个，只剩下75，而后面覆盖了后面一个帧的空载波和训练序列的CP（为什么最后要减去1，因为要的是最后一个的序号，而序号位置是点数减去1才对，97到279135的个数是279135-97+1）
    edge_index = 1;
    % 符号同步，利用Long training周期性，进一步求得符号开始的精确时刻(比起packet_edge算是细同步)
    [fine_time_est,phase_data(loop+1),max_peak_long]= rx_fine_time_sync_cross_corr(rx_signal,LtrsCPLength,LongTrainSym,zeros_head,plot_en);

    fine_index(loop+1) = fine_index(loop)+symbol_bits+fine_time_est+edge_index-(zeros_head+LtrsCPLength); %下一帧同步的大概位置
%     fine_index(loop+1) = fine_index(loop)+symbol_bits+fine_time_est+edge_index-2-(zeros_head+LtrsCPLength); %下一帧同步的大概位置
%     fine_index(loop) = fine_index_temp+fine_time_est+edge_index-(zeros_head+LtrsCPLength-5);%本帧同步的精确位置
    %       fine_index(loop+1) = fine_index(loop)+symbol_bits+fine_time_est+edge_index-(zeros_head+CPLength); %下一帧同步的大概位置

    %      fine_index(loop) = fine_index_temp+fine_time_est+edge_index-(zeros_head+CPLength);%本帧同步的精确位置
      % fine_time_est=92;

    syn2_time_signal = rx_signal(fine_time_est: length(rx_signal)); % 经过精确同步后的本帧头部位置确定下来，后面的长度仍然包括下一个帧的,包括两个训练序列和后续的数据及其CP

    % 频偏估计
    [freq_offset, syn3_signal] = freq_offset_esti(syn2_time_signal, Fg, NumberOfIFFTSamples, Ltrs_num);

    % FFT 还原数据，分开数据和导频
    [freq_tr_syms, freq_data_syms] = rx_FFT(syn2_time_signal,NumberOfIFFTSamples,NumberOfGuardTime,sizeofqamsymbol,Ltrs_num);

  SRX= freq_data_syms;
    SignalTimeIX=real(SRX);
SignalTimeQX=imag(SRX);




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

SpectralrcX   = SpectralIQX.';
% 信道估计，利用FFT还原后的频域长训练序列，除以已知的长训练序列
    channel_est = channel_esti(freq_tr_syms,[1 129],LongTrainSym,Ltrs_num);
    % 信道均衡

    freq_data_syms = rx_channel_equal(SpectralrcX, channel_est, sizeofqamsymbol);
    % freq_dfreq_data_symsata_syms1=reshape(freq_data_syms,[],1);
    figure, plot(freq_data_syms,'b.'), title('均衡后'),axis(1.5*[-1 1,-1 1]);


    % [data_bps_out, bps_out] = bps_(freq_data_syms,32,10,qammod(0:3,4), pi/2,sizeofqamsymbol,NumberOfIFFTSamples,[1 129],1);
    % freq_data_syms_out=[freq_data_syms_out; data_bps_out];
%% did fftshift before ifft, need to do fftshift again in the receiver
IQXmatrix = zeros(NrOfOFDMSymbols,NumberOfCarriers);
freq_data_syms_out=freq_data_syms_out.';
for i = 1:NrOfOFDMSymbols
    IQXmatrix(i , :)                     = freq_data_syms(i , Carrier_location);
end
%% Normalization

SRX_deofdm = reshape(IQXmatrix,1,size(IQXmatrix,1)*size(IQXmatrix,2));
end


end

