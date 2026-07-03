% Channel estimation 
% 最小二乘法估计 LS  adding by min
function channel_estimate = channel_esti(freq_tr_syms,NullsubcInd,LongTrainSym,Ltrs_num)   
	LongTrainSym(NullsubcInd)=1;
	if Ltrs_num==1
        mean_symbols = freq_tr_syms;
    else
        mean_symbols = mean(freq_tr_syms);
    end
	channel_estimate = mean_symbols ./ LongTrainSym;  % 频域/频域，频域上的信道增益
	channel_estimate(NullsubcInd) = 0;  
end