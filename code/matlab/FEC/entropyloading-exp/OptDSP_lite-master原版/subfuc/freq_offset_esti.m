function [freq_offset, data] = freq_offset_esti(ofdm_signal, fs, numFFT, Ltrs_num)

long_training_end = numFFT*Ltrs_num;  % long_training_end = 64*1

% Long Training symbols
long_tr_syms = ofdm_signal(1: long_training_end);        % 本帧数据中的长训练序列
long_tr_syms_1 = conj(long_tr_syms(1:numFFT));   % 把longtrain分为两行(本来就是两个周期重复的长训练序列)
long_tr_syms_2 = long_tr_syms(numFFT + 1 : end); % matrix size:1×64
R = sum(long_tr_syms_1.*long_tr_syms_2);
phase = angle(R);
delta_freq = fs/numFFT;
freq_offset = phase * delta_freq /(2* pi);
fprintf('The estimated frequency offset is %4.1f kHz \n',freq_offset/1000); 
t=(0:length(ofdm_signal)-1)./fs;
data = ofdm_signal.*exp(-1i*2*pi*freq_offset*t);
end