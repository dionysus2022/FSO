function [GMI,NGMI] = runonce(SNR_dB,nSpS,SIG,TX)



TX.BIT.txBits = Tx_generateBits(SIG.nSyms,TX.QAM.M,TX.QAM.nPol,TX.BIT);


[S.tx,txSyms,TX.QAM] = Tx_ProbShaping(TX.BIT.txBits,TX.QAM,TX.SIG,TX.FEC.rate);

S.txSC = upsample(S.tx,nSpS);

S.rx = setSNR(S.txSC,SNR_dB,TX.DAC.RESAMP.sampRate,SIG.symRate);%这一步是真的加了噪声的


S.rx_1sps = S.rx(1:nSpS:end);%下采样，直接隔一个删一个
DSP.DEMAPPER.normMethod = 'MMSE';
C = TX.QAM.IQmap;

[DSP.DEMAPPER,S.tx] = symDemapper(S.rx_1sps,S.tx,C,DSP.DEMAPPER);%注意这里s.tx是用满足MMSE的系数乘过的。已判决完和比特

[GMI,NGMI] = GMI_eval(S.rx_1sps,DSP.DEMAPPER.txBits,DSP.DEMAPPER.C,DSP.DEMAPPER.N0,TX.QAM.symProb);





end

