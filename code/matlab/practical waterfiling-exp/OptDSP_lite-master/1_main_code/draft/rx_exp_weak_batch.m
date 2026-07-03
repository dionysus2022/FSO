% rx_exp_weak_batch.m - batch RX with deep fade detect & 2D heatmap
clear; clear global; close all; clc;

addpath(genpath('D:\matlab\OptDSP_FSO_2\practical waterfiling-exp\OptDSP_lite-master\1_main_code'));
global PROG; PROG.showMessagesLevel = 2; initProg(); RGB = fancyColors(); co=1;

SIG.M=4; SIG.symRate=8e9/co; SIG.bitRate_net=8e9; SIG.modulation='QAM';
SIG.rollOff=0.25; SIG.nPol=1; SIG.nSyms=2^7/co; nSpS=5; laserLW=0e6;
FEC_rate=1; pilotRate=1; useCPE2=false; SNR_dB=80;

ofdm.NumberOfIFFTSamples=256; ofdm.Carrier_location=[4:126];
ofdm.Carrier_location_demo=[4:126,132:254];
ofdm.NumberOfCarriers=length(ofdm.Carrier_location);
ofdm.NumberOfCarriers_demo=length(ofdm.Carrier_location_demo);
ofdm.NumberOfGuardTime=16; Fs=10e9; Fg=10e9;
nBpS_net=SIG.bitRate_net/(SIG.nPol*SIG.symRate*FEC_rate*pilotRate);
TX.SIG=setSignalParams('symRate',SIG.symRate,'M',SIG.M,...
    'nPol',SIG.nPol,'nBpS',nBpS_net,'nSyms',SIG.nSyms,...
    'roll-off',SIG.rollOff,'modulation',SIG.modulation);
TX.QAM=QAM_config(TX.SIG); TX.BIT.source='randi'; TX.BIT.seed=100;
TX.PS.type='RRC'; TX.PS.rollOff=TX.SIG.rollOff; TX.PS.nTaps=4096;
TX.DAC.RESAMP.sampRate=nSpS*TX.SIG.symRate; TX.LASER.linewidth=laserLW;
TX.PILOTS.active=true; TX.PILOTS.rate=pilotRate; TX.PILOTS.option='outerQPSK';
TX.FEC.active=false; TX.FEC.rate=FEC_rate; TX.FEC.nIter=50; TX.PCS.method='CCDM';
ofdm.size=SIG.nSyms;

set(0,'DefaultFigureVisible','off');
data_root='D:\matlab\OptDSP_FSO_2\practical waterfiling-exp\OptDSP_lite-master\2_Data_Results';
save_dir=fullfile(data_root,'figures');
if ~exist(save_dir,'dir'), mkdir(save_dir); end

mod_name='2QAM';
rx_dir=fullfile(data_root,'rx_data',mod_name);
ref_dir=fullfile(data_root,'tx_1frame_5mod',mod_name);
t_start=1; t_end=75; scope_Fs=80e9; SNR_fade_thresh=3; N_per_sub=25;

DSP.MF.type='RRC'; DSP.MF.rollOff=TX.SIG.rollOff;
DSP.CPE1.method='pilot-based:optimized'; DSP.CPE1.decision='data-aided';
DSP.CPE1.nTaps_min=1; DSP.CPE1.nTaps_max=201; DSP.CPE1.PILOTS=TX.PILOTS;
DSP.CPE2.method='BPS'; DSP.CPE2.nTaps=22;
DSP.CPE2.nTaps_min=1; DSP.CPE2.nTaps_max=501;
DSP.CPE2.nTestPhases=10; DSP.CPE2.angleInterval=pi/8;
DSP.DEMAPPER.normMethod='MMSE'; C=TX.QAM.IQmap;

N_sc=ofdm.NumberOfCarriers;
all_snr=NaN(N_sc,t_end); valid_snr=[]; valid_idx=[];

for t=t_start:t_end
    fprintf('\n--- Group %d/%d ---\n',t,t_end);
    sub=ceil(t/N_per_sub);
    fn=fullfile(rx_dir,num2str(sub),sprintf('%d.bin',t));
    if ~exist(fn,'file'), fprintf('  No file\n'); continue; end
    d=dir(fn);
    if d.bytes==0, fprintf('  Empty\n'); continue; end

    fid=fopen(fn,'rb');
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

    ref_fn=fullfile(ref_dir,sprintf('sub%02d',sub),sprintf('sig_%04d.mat',t));
    if ~exist(ref_fn,'file'), fprintf('  No ref\n'); continue; end
    load(ref_fn,'data_tx');

    s=resample(s,16e9,scope_Fs);
    s=s-mean(s);
    s=s*(1/(sum(abs(s))/length(s)));

    rx=deOFDM(s,ofdm,SIG.nSyms);
    tx=data_tx.';
    rx=reshape(rx,SIG.nSyms,ofdm.NumberOfCarriers_demo).';

    for i=1:N_sc
        [DSP.DEMAPPER,td]=symDemapper(rx(i,:),tx(i,:),C,DSP.DEMAPPER);
        tm(i,:)=td; DSP.DEMAPPER.N0=0;
    end

    [~,SNR]=EVM_eval(rx(1:N_sc,:),tm);
    v=SNR(SNR>0&isfinite(SNR));
    a=10*log10(mean(10.^(v/10)));

    if a<SNR_fade_thresh||isnan(a)
        fprintf('  Deep fade (%.1f dB)\n',a); continue;
    end

    all_snr(:,t)=SNR; valid_snr=[valid_snr;a]; valid_idx=[valid_idx;t];
    fprintf('  SNR = %.2f dB\n',a);
end

%% Stats
fprintf('\n=== Scintillation ===\n');
if length(valid_snr)>=2
    fprintf('  Valid: %d/%d\n',length(valid_snr),t_end);
    fprintf('  Mean SNR: %.2f dB\n',mean(valid_snr));
    fprintf('  SNR var: %.4f, std: %.4f\n',var(valid_snr),std(valid_snr));
    li=10.^(valid_snr/10);
    fprintf('  Scintillation index: %.4f\n',var(li)/mean(li)^2);
end

%% 2D map
vc=any(~isnan(all_snr),1);
fh=figure('Visible','off');
imagesc(find(vc),1:N_sc,all_snr(:,vc));
colorbar; caxis([0 30]);
xlabel('Group t'); ylabel('Subcarrier'); title('SNR fading map');
set(gca,'YDir','normal'); colormap(jet);
saveas(fh,fullfile(save_dir,'snr_map.png'));
save(fullfile(save_dir,'batch_results.mat'),'all_snr','valid_snr','valid_idx');
fprintf('Saved\nDone\n');
