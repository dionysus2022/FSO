% rx_exp_weak_sub1.m - process all files in first subfolder, save results
clear; clear global; close all; clc;

addpath(genpath('D:\matlab\OptDSP_FSO_2\practical waterfiling-exp\OptDSP_lite-master'));
global PROG; PROG.showMessagesLevel = 2; initProg(); RGB = fancyColors(); co=1;

SIG.M = 4; SIG.symRate = 8e9/co; SIG.bitRate_net = 8e9;
SIG.modulation = 'QAM'; SIG.rollOff = 0.25; SIG.nPol = 1;
SIG.nSyms = 2^7/co; nSpS = 5; laserLW = 0e6;
FEC_rate = 1; pilotRate = 1; useCPE2 = false; SNR_dB = 80;

ofdm.NumberOfIFFTSamples=256; ofdm.Carrier_location=[4:126];
ofdm.Carrier_location_demo=[4:126,132:254];
ofdm.NumberOfCarriers=length(ofdm.Carrier_location);
ofdm.NumberOfCarriers_demo=length(ofdm.Carrier_location_demo);
ofdm.NumberOfGuardTime=16; Fs=10e9; Fg=10e9;

nBpS_net = SIG.bitRate_net/(SIG.nPol*SIG.symRate*FEC_rate*pilotRate);
TX.SIG = setSignalParams('symRate',SIG.symRate,'M',SIG.M,...
    'nPol',SIG.nPol,'nBpS',nBpS_net,'nSyms',SIG.nSyms,...
    'roll-off',SIG.rollOff,'modulation',SIG.modulation);
TX.QAM = QAM_config(TX.SIG); TX.BIT.source = 'randi'; TX.BIT.seed = 100;
TX.PS.type = 'RRC'; TX.PS.rollOff = TX.SIG.rollOff; TX.PS.nTaps = 4096;
TX.DAC.RESAMP.sampRate = nSpS*TX.SIG.symRate; TX.LASER.linewidth = laserLW;
TX.PILOTS.active = true; TX.PILOTS.rate = pilotRate; TX.PILOTS.option = 'outerQPSK';
TX.FEC.active = false; TX.FEC.rate = FEC_rate; TX.FEC.nIter = 50; TX.PCS.method = 'CCDM';
ofdm.size = SIG.nSyms;

C = TX.QAM.IQmap;
DSP.MF.type='RRC'; DSP.MF.rollOff=TX.SIG.rollOff;
DSP.CPE1.method='pilot-based:optimized'; DSP.CPE1.decision='data-aided';
DSP.CPE1.nTaps_min=1; DSP.CPE1.nTaps_max=201; DSP.CPE1.PILOTS=TX.PILOTS;
DSP.CPE2.method='BPS'; DSP.CPE2.nTaps=22;
DSP.CPE2.nTaps_min=1; DSP.CPE2.nTaps_max=501;
DSP.CPE2.nTestPhases=10; DSP.CPE2.angleInterval=pi/8;
DSP.DEMAPPER.normMethod='MMSE';

%% Config
data_root = 'D:\matlab\OptDSP_FSO_2\practical waterfiling-exp\OptDSP_lite-master\2_Data_Results';
mod_name = '2QAM';
rx_dir = fullfile(data_root,'rx_data',mod_name,'1');     % first subfolder
ref_dir = fullfile(data_root,'tx_1frame_5mod',mod_name,'sub01');
save_dir = fullfile(data_root,'results',mod_name,'sub01');
if ~exist(save_dir,'dir'), mkdir(save_dir); end

t_start = 1; t_end = 25; scope_Fs = 80e9;
N_sc = ofdm.NumberOfCarriers;

all_snr = NaN(N_sc,t_end);
all_avg = NaN(t_end,1);

for t = t_start:t_end
    fprintf('\n--- File %d/%d ---\n',t,t_end);

    bin_file = fullfile(rx_dir,sprintf('%d.bin',t));
    mat_file = fullfile(ref_dir,sprintf('sig_%04d.mat',t));

    if ~exist(bin_file,'file'), fprintf('  No file\n'); continue; end
    d = dir(bin_file);
    if d.bytes==0, fprintf('  Empty\n'); continue; end
    if ~exist(mat_file,'file'), fprintf('  No ref\n'); continue; end

    %% 1. Read .bin
    fid = fopen(bin_file,'rb');
    fread(fid,2,'*char')'; fread(fid,2,'*char')';
    fread(fid,1,'int32'); fread(fid,1,'int32');
    fread(fid,1,'int32'); fread(fid,1,'int32');
    fread(fid,1,'int32'); fread(fid,1,'int32');
    np=fread(fid,1,'int32');
    fread(fid,1,'int32'); fread(fid,1,'float32');
    fread(fid,1,'float64'); fread(fid,1,'float64');
    fread(fid,1,'float64'); fread(fid,1,'int32');
    fread(fid,1,'int32'); fread(fid,16,'*char')';
    fread(fid,16,'*char')'; fread(fid,24,'*char')';
    fread(fid,16,'*char')'; fread(fid,1,'float64');
    fread(fid,1,'uint32'); fread(fid,1,'int32');
    fread(fid,1,'int16'); bpp=fread(fid,1,'int16');
    fread(fid,1,'int32');
    if isempty(bpp)||~isscalar(bpp), fclose(fid); fprintf('  Bad header\n'); continue; end
    switch bpp
        case 4, s=fread(fid,np,'float32').';
        case 2, s=fread(fid,np,'int16').';
        case 1, s=fread(fid,np,'int8').';
        otherwise, s=fread(fid,np,'double').';
    end
    fclose(fid);

    %% 2. Load ref
    load(mat_file,'data_tx');

    %% 3. Resample + normalize
    s = resample(s,16e9,scope_Fs);
    s = s - mean(s);
    s = s * (1/(sum(abs(s))/length(s)));

    %% 4. deOFDM + demap
    S.rx_1sps = s;
    S.rx_1sps = deOFDM(S.rx_1sps,ofdm,SIG.nSyms);
    S.tx = data_tx.';
    S.rx_1sps = reshape(S.rx_1sps,SIG.nSyms,ofdm.NumberOfCarriers_demo).';

    for i=1:N_sc
        [DSP.DEMAPPER,td]=symDemapper(S.rx_1sps(i,:),S.tx(i,:),C,DSP.DEMAPPER);
        tm(i,:)=td; DSP.DEMAPPER.N0=0;
    end

    [~,SNR]=EVM_eval(S.rx_1sps(1:N_sc,:),tm);
    v=SNR(SNR>0&isfinite(SNR));
    a=10*log10(mean(10.^(v/10)));

    if a<3||isnan(a)
        fprintf('  Deep fade (%.1f dB)\n',a); continue;
    end

    all_snr(:,t)=SNR; all_avg(t)=a;
    fprintf('  SNR = %.2f dB\n',a);
end

%% Save results
save(fullfile(save_dir,'snr_results.mat'),'all_snr','all_avg');
fprintf('\nResults saved to %s\n',save_dir);
fprintf('Done\n');
