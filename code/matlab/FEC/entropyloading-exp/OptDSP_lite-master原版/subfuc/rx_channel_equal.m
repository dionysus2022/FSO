%Channel equalization
function data_syms_out = rx_channel_equal(freq_data_syms, channel_est, numsym)   

    % B = repmat(A,M,N) creates a large matrix B consisting of an M-by-N  tiling of copies of A
    % channel_est:1*64  data_syms_out = freq_data_syms ./ chan_corr_mat;
    chan_corr_mat = repmat(channel_est, numsym, 1); % 把channel_est复制NumOFDMSymbols这么多行    
   % data_syms_out = freq_data_syms ./ chan_corr_mat;
      data_syms_out = freq_data_syms ./(chan_corr_mat) ;
%     Amplitude normalization
%     chan_sq_amplitude = sum(abs(channel_est(ofdm_data_parm_var.DataSubcIdx)) .^ 2);
%     data_syms_out = data_syms_out / chan_sq_amplitude;
end