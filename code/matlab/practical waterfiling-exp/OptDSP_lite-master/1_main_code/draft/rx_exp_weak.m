% rx_exp_weak.m - single RX, reads Keysight .bin
%D:\Project_code\OptDSP_FSO_2\practical waterfiling-exp\OptDSP_lite-master\1_main_code
clear; clear global; close all; clc;

addpath(genpath('D:\Project_code\OptDSP_FSO_2\practical waterfiling-exp\OptDSP_lite-master'));
global PROG; PROG.showMessagesLevel = 2; initProg(); RGB = fancyColors(); co=1;

SIG.M = 256; SIG.symRate = 8e9/co; SIG.bitRate_net = 8e9;
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

%% File paths
rx_bin_file = 'D:\Project_code\OptDSP_FSO_2\practical waterfiling-exp\OptDSP_lite-master\2_Data_Results\rx_data\2026.06.27\256QAM\sub01\2.bin';
ref_mat_file = 'D:\Project_code\OptDSP_FSO_2\practical waterfiling-exp\OptDSP_lite-master\2_Data_Results\tx_1frame_6mod_128sym\256QAM\sub01\sig_0002.mat';
scope_Fs = 80e9;

%% 1. Read Keysight .bin
fprintf('\n[1] Reading scope .bin...\n');
fid = fopen(rx_bin_file,'rb');
if fid == -1, error('Cannot open'); end
cookie=fread(fid,2,'*char')'; version=fread(fid,2,'*char')';
file_size=fread(fid,1,'int32'); num_waveforms=fread(fid,1,'int32');
header_size=fread(fid,1,'int32'); wave_type=fread(fid,1,'int32');
num_buffers=fread(fid,1,'int32'); num_points=fread(fid,1,'int32');
count=fread(fid,1,'int32'); x_disp_range=fread(fid,1,'float32');
x_disp_orig=fread(fid,1,'float64'); x_inc=fread(fid,1,'float64');
x_orig=fread(fid,1,'float64'); x_units=fread(fid,1,'int32');
y_units=fread(fid,1,'int32'); date_str=fread(fid,16,'*char')';
time_str=fread(fid,16,'*char')'; frame_str=fread(fid,24,'*char')';
wave_str=fread(fid,16,'*char')'; time_tag=fread(fid,1,'float64');
segment_index=fread(fid,1,'uint32'); data_header_size=fread(fid,1,'int32');
buffer_type=fread(fid,1,'int16'); bytes_per_point=fread(fid,1,'int16');
buffer_size=fread(fid,1,'int32');
fprintf('  Time: %s %s, Points: %d, %d bytes/pt\n',date_str,time_str,num_points,bytes_per_point);
switch bytes_per_point
    case 4, OutputFSO=fread(fid,num_points,'float32').';
    case 2, OutputFSO=fread(fid,num_points,'int16').';
    case 1, OutputFSO=fread(fid,num_points,'int8').';
    otherwise, OutputFSO=fread(fid,num_points,'double').';
end
fclose(fid);

%% 2. Load reference
load(ref_mat_file);

%% 3. DSP setup
DSP.MF.type='RRC'; DSP.MF.rollOff=TX.SIG.rollOff;
DSP.CPE1.method='pilot-based:optimized'; DSP.CPE1.decision='data-aided';
DSP.CPE1.nTaps_min=1; DSP.CPE1.nTaps_max=201; DSP.CPE1.PILOTS=TX.PILOTS;
DSP.CPE2.method='BPS'; DSP.CPE2.nTaps=22;
DSP.CPE2.nTaps_min=1; DSP.CPE2.nTaps_max=501;
DSP.CPE2.nTestPhases=10; DSP.CPE2.angleInterval=pi/8;
DSP.DEMAPPER.normMethod='MMSE';

%% 4. Resample
fprintf('[2] Resample %.0f GS/s -> 16 GS/s\n',scope_Fs/1e9);
OutputFSO=resample(OutputFSO,16e9,scope_Fs);
data_in_mean2=mean(OutputFSO);
OutputFSO=OutputFSO-data_in_mean2;
data_in_Amp2=sum(abs(OutputFSO))/length(OutputFSO);
AMP_rate2=1/data_in_Amp2;
data_normal2=OutputFSO*AMP_rate2;
data_in=data_normal2;

%% 5. deOFDM
fprintf('[3] deOFDM\n');
S.rx_1sps=data_in;
S.rx_1sps=deOFDM(S.rx_1sps,ofdm,SIG.nSyms);
if pilotRate<1, [S.rx_1sps,DSP.CPE1]=carrierPhaseEstimation(S.rx_1sps,S.tx,DSP.CPE1); end
C=TX.QAM.IQmap;
if useCPE2, [S.rx_1sps,DSP.CPE2]=carrierPhaseEstimation(S.rx_1sps,S.tx,DSP.CPE2,C); end
if pilotRate<1, [S.rx_1sps,S.tx]=pilotSymbols_rmv(S.rx_1sps,S.tx,DSP.CPE1.PILOTS); end
S.tx=data_tx.';
S.rx_1sps=reshape(S.rx_1sps,SIG.nSyms,ofdm.NumberOfCarriers_demo);
S.rx_1sps=S.rx_1sps.';

%% 6. Demapper + SNR
for i=1:123
    [DSP.DEMAPPER,S.txafdem]=symDemapper(S.rx_1sps(i,:),S.tx(i,:),C,DSP.DEMAPPER);
    [BER,~]=BER_eval(DSP.DEMAPPER.txBits,DSP.DEMAPPER.rxBits);
    S.BER(i,:)=BER; DSP.DEMAPPER.N0=0;
    S.txafdem_matrix(i,:)=S.txafdem;
end
BERMEAN=mean(S.BER);
S.rx_1sps=S.rx_1sps(1:123,:);
[EVM,SNR_CAL]=EVM_eval(S.rx_1sps,S.txafdem_matrix);

figure(); plot(SNR_CAL);
v=SNR_CAL(SNR_CAL>0&isfinite(SNR_CAL));
avg_SNR=10*log10(mean(10.^(v/10)));
fprintf('\nAvg SNR: %.2f dB\n',avg_SNR);

fprintf('Done\n');
