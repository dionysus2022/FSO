% rx_exp_weak_6mod.m - batch RX processing for 6 modulations (QPSK,16QAM,32QAM,64QAM,128QAM,256QAM)
% 平均snr计算
clear; clear global; close all; clc;

addpath(genpath('D:\Project_code\OptDSP_FSO_2\practical waterfiling-exp\OptDSP_lite-master'));
global PROG; PROG.showMessagesLevel = 2; initProg(); RGB = fancyColors(); co=1;

%% Common parameters
SIG.symRate = 8e9/co; SIG.bitRate_net = 8e9;
SIG.modulation = 'QAM'; SIG.rollOff = 0.25; SIG.nPol = 1;
SIG.nSyms = 2^7/co; nSpS = 5; laserLW = 0e6;
FEC_rate = 1; pilotRate = 1; useCPE2 = false; SNR_dB = 80;

ofdm.NumberOfIFFTSamples=256; ofdm.Carrier_location=[4:126];
ofdm.Carrier_location_demo=[4:126,132:254];
ofdm.NumberOfCarriers=length(ofdm.Carrier_location);
ofdm.NumberOfCarriers_demo=length(ofdm.Carrier_location_demo);
ofdm.NumberOfGuardTime=16; Fs=10e9; Fg=10e9;

nBpS_net = SIG.bitRate_net/(SIG.nPol*SIG.symRate*FEC_rate*pilotRate);

%% Modulation config - match tx_1frame_6mod.m
mod_names = {'QPSK','16QAM','32QAM','64QAM','128QAM','256QAM'};
mod_M     = [4,      16,     32,     64,     128,     256];

%% Paths
rx_root = 'D:\Project_code\OptDSP_FSO_2\practical waterfiling-exp\OptDSP_lite-master\2_Data_Results\rx_data\2026.06.27';
ref_root = 'D:\Project_code\OptDSP_FSO_2\practical waterfiling-exp\OptDSP_lite-master\2_Data_Results\tx_1frame_6mod_128sym';

out_root = 'D:\Project_code\OptDSP_FSO_2\practical waterfiling-exp\OptDSP_lite-master\2_Data_Results\rx_exp_weak_6mod_results';
if ~exist(out_root,'dir'), mkdir(out_root); end

%% Results storage
results = {};

for m = 1:length(mod_names)
    mname = mod_names{m};
    M = mod_M(m);
    fprintf('\n========================================\n');
    fprintf('=== %s (M=%d) ===\n', mname, M);
    fprintf('========================================\n');

    % --- Setup per-modulation TX params ---
    SIG.M = M;
    TX.SIG = setSignalParams('symRate',SIG.symRate,'M',SIG.M,...
        'nPol',SIG.nPol,'nBpS',nBpS_net,'nSyms',SIG.nSyms,...
        'roll-off',SIG.rollOff,'modulation',SIG.modulation);
    TX.QAM = QAM_config(TX.SIG);
    TX.BIT.source = 'randi'; TX.BIT.seed = 100;
    TX.PS.type = 'RRC'; TX.PS.rollOff = TX.SIG.rollOff; TX.PS.nTaps = 4096;
    TX.DAC.RESAMP.sampRate = nSpS*TX.SIG.symRate; TX.LASER.linewidth = laserLW;
    TX.PILOTS.active = true; TX.PILOTS.rate = pilotRate; TX.PILOTS.option = 'outerQPSK';
    TX.FEC.active = false; TX.FEC.rate = FEC_rate; TX.FEC.nIter = 50; TX.PCS.method = 'CCDM';
    ofdm.size = SIG.nSyms;

    % --- DSP setup (fixed per modulation) ---
    DSP.MF.type='RRC'; DSP.MF.rollOff=TX.SIG.rollOff;
    DSP.CPE1.method='pilot-based:optimized'; DSP.CPE1.decision='data-aided';
    DSP.CPE1.nTaps_min=1; DSP.CPE1.nTaps_max=201; DSP.CPE1.PILOTS=TX.PILOTS;
    DSP.CPE2.method='BPS'; DSP.CPE2.nTaps=22;
    DSP.CPE2.nTaps_min=1; DSP.CPE2.nTaps_max=501;
    DSP.CPE2.nTestPhases=10; DSP.CPE2.angleInterval=pi/8;
    DSP.DEMAPPER.normMethod='MMSE';

    % --- Get sub-folders list from rx_data ---
    rx_md = fullfile(rx_root, mname);
    if ~exist(rx_md,'dir')
        fprintf('  WARNING: %s not found, skipping\n', rx_md);
        continue;
    end
    sub_dirs = dir(rx_md);
    sub_dirs = sub_dirs([sub_dirs.isdir]);
    sub_dirs = sub_dirs(~ismember({sub_dirs.name},{'.','..'}));
    sub_names = sort({sub_dirs.name});

    avg_SNR_all = [];

    for si = 1:length(sub_names)
        sub_name = sub_names{si};
        bin_dir = fullfile(rx_md, sub_name);
        ref_dir = fullfile(ref_root, mname, sub_name);

        bin_files = dir(fullfile(bin_dir, '*.bin'));
        bin_names = sort({bin_files.name});

        if ~exist(ref_dir,'dir')
            fprintf('  WARNING: ref dir %s not found, skipping\n', ref_dir);
            continue;
        end

        for bi = 1:length(bin_names)
            [~, bin_stem, ~] = fileparts(bin_names{bi});
            bin_file = fullfile(bin_dir, bin_names{bi});
            ref_file = fullfile(ref_dir, sprintf('sig_%04d.mat', str2double(bin_stem)));

            if ~exist(ref_file,'file')
                fprintf('  WARNING: ref file %s not found, skipping\n', ref_file);
                continue;
            end

            fprintf('\n[%s %s %s]\n', mname, sub_name, bin_names{bi});

            % ---- 1. Read .bin ----
            fid = fopen(bin_file,'rb');
            if fid == -1
                fprintf('  ERROR: cannot open %s\n', bin_file);
                continue;
            end
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
            if isempty(bytes_per_point)
                fprintf('  WARNING: empty bytes_per_point, skipping file\n');
                fclose(fid); continue;
            end
            bytes_per_point = bytes_per_point(1);  % ensure scalar
            switch bytes_per_point
                case 4, OutputFSO=fread(fid,num_points,'float32').';
                case 2, OutputFSO=fread(fid,num_points,'int16').';
                case 1, OutputFSO=fread(fid,num_points,'int8').';
                otherwise, OutputFSO=fread(fid,num_points,'double').';
            end
            fclose(fid);

            % ---- 2. Load reference ----
            load(ref_file);  % loads data_tx

            % ---- 3. Resample ----
            scope_Fs = 80e9;
            OutputFSO=resample(OutputFSO,16e9,scope_Fs);
            data_in_mean2=mean(OutputFSO);
            OutputFSO=OutputFSO-data_in_mean2;
            data_in_Amp2=sum(abs(OutputFSO))/length(OutputFSO);
            AMP_rate2=1/data_in_Amp2;
            data_normal2=OutputFSO*AMP_rate2;
            data_in=data_normal2;

            % ---- 4. deOFDM + Demapper + SNR (wrapped in try-catch) ----
            try
                S.rx_1sps=data_in;
                S.rx_1sps=deOFDM(S.rx_1sps,ofdm,SIG.nSyms);
                if pilotRate<1, [S.rx_1sps,DSP.CPE1]=carrierPhaseEstimation(S.rx_1sps,S.tx,DSP.CPE1); end
                C=TX.QAM.IQmap;
                if useCPE2, [S.rx_1sps,DSP.CPE2]=carrierPhaseEstimation(S.rx_1sps,S.tx,DSP.CPE2,C); end
                if pilotRate<1, [S.rx_1sps,S.tx]=pilotSymbols_rmv(S.rx_1sps,S.tx,DSP.CPE1.PILOTS); end
                S.tx=data_tx.';
                S.rx_1sps=reshape(S.rx_1sps,SIG.nSyms,ofdm.NumberOfCarriers_demo);
                S.rx_1sps=S.rx_1sps.';

                % ---- 5. Demapper + SNR ----
                for i=1:123
                    [DSP.DEMAPPER,S.txafdem]=symDemapper(S.rx_1sps(i,:),S.tx(i,:),C,DSP.DEMAPPER);
                    [BER,~]=BER_eval(DSP.DEMAPPER.txBits,DSP.DEMAPPER.rxBits);
                    S.BER(i,:)=BER; DSP.DEMAPPER.N0=0;
                    S.txafdem_matrix(i,:)=S.txafdem;
                end
                BERMEAN=mean(S.BER);
                S.rx_1sps=S.rx_1sps(1:123,:);
                [EVM,SNR_CAL]=EVM_eval(S.rx_1sps,S.txafdem_matrix);

                v=SNR_CAL(SNR_CAL>0&isfinite(SNR_CAL));
                avg_SNR=10*log10(mean(10.^(v/10)));
                avg_SNR_all(end+1) = avg_SNR;

                fprintf('  Avg SNR: %.2f dB\n', avg_SNR);

                % Store result
                results(end+1,:) = {mname, sub_name, bin_stem, avg_SNR};

                figure(); plot(SNR_CAL);
            catch ME
                fprintf('  WARNING: processing failed (%s), skipping\n', ME.message);
            end
        end
    end

    % ---- Summary for this modulation ----
    fprintf('\n--- %s Summary ---\n', mname);
    if ~isempty(avg_SNR_all)
        fprintf('  Files processed: %d\n', length(avg_SNR_all));
        fprintf('  Mean SNR: %.2f dB\n', mean(avg_SNR_all));
        fprintf('  Min SNR:  %.2f dB\n', min(avg_SNR_all));
        fprintf('  Max SNR:  %.2f dB\n', max(avg_SNR_all));
        fprintf('  Std SNR:  %.2f dB\n', std(avg_SNR_all));
    else
        fprintf('  No files processed\n');
    end

    % Save per-modulation intermediate result
    if ~isempty(avg_SNR_all)
        save(fullfile(out_root, sprintf('results_%s.mat', mname)), 'avg_SNR_all');
    end
end

%% ---- Final summary ----
fprintf('\n\n========================================\n');
fprintf('=== FINAL SUMMARY ===\n');
fprintf('========================================\n');
if ~isempty(results)
    mods_processed = unique(results(:,1));
    for m = 1:length(mods_processed)
        mn = mods_processed{m};
        idx = strcmp(results(:,1), mn);
        snr_vals = cell2mat(results(idx,4));
        fprintf('  %s: %d files, mean SNR = %.2f dB\n', mn, length(snr_vals), mean(snr_vals));
    end
    % Save full results
    save(fullfile(out_root, 'results_all.mat'), 'results');
    fprintf('\nFull results saved to %s\n', fullfile(out_root, 'results_all.mat'));
else
    fprintf('  No results\n');
end

fprintf('\nAll done!\n');