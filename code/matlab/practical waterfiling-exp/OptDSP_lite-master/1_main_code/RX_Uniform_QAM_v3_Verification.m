
%% ============================================================
%  RX Uniform QAM v3 Verification
%  对应 TX_uniform_QAM_batch_v3
% ============================================================

clear; clc;

%% ========== CONFIG ==========
cfg.n_sc = 64;
cfg.n_sym = 32;
cfg.cp_len = 16;

cfg.mods = {'QPSK','16QAM','32QAM','64QAM','128QAM','256QAM'};

cfg.root_dir = './TX_UNIFORM_BATCH_V3/';

cfg.subfolders = {'sub01','sub03'};

%% ========== RESULT STORAGE ==========
result = struct();

%% ============================================================
for s = 1:length(cfg.subfolders)

    sub = cfg.subfolders{s};

    for m = 1:length(cfg.mods)

        mod_name = cfg.mods{m};

        in_dir = fullfile(cfg.root_dir, sub, mod_name);

        files = dir(fullfile(in_dir,'*.bin'));

        fprintf('\n[%s - %s] processing %d files\n', sub, mod_name, length(files));

        constellation_all = [];

        snr_list = [];

        for i = 1:length(files)

            %% ===== load TX signal =====
            fid = fopen(fullfile(in_dir, files(i).name), 'rb');
            raw = fread(fid, 'float32');
            fclose(fid);

            tx = raw(1:2:end) + 1j * raw(2:2:end);

            %% ===== reshape into frames =====
            frame_len = (cfg.n_sc+cfg.cp_len)*cfg.n_sym;
            n_frame = length(tx) / frame_len;

            for f = 1:n_frame

                idx = (f-1)*frame_len + (1:frame_len);
                x = tx(idx);

                %% ===== remove CP =====
                x = reshape(x, cfg.n_sc+cfg.cp_len, cfg.n_sym);
                x = x(cfg.cp_len+1:end, :);

                %% ===== FFT =====
                X = fft(x, [], 1);

                %% ===== flatten =====
                Xvec = X(:);

                constellation_all = [constellation_all; Xvec];

                %% ===== SNR estimation =====
                noise = Xvec - mean(Xvec);
                snr_est = 10*log10(mean(abs(Xvec).^2) / mean(abs(noise).^2 + 1e-12));
                snr_list = [snr_list; snr_est];

            end
        end

        %% ========================================================
        %  1. CONSTELLATION VISUALIZATION
        %% ========================================================
        figure;
        scatter(real(constellation_all), imag(constellation_all), 2, '.');
        title([sub ' - ' mod_name ' RX Constellation']);
        axis equal; grid on;

        %% ========================================================
        %  2. NORMALIZATION (for comparison only)
        %% ========================================================
        Xn = constellation_all / sqrt(mean(abs(constellation_all).^2));

        %% ========================================================
        %  3. STATISTICS
        %% ========================================================
        result.(sub).(mod_name).mean_power = mean(abs(Xn).^2);
        result.(sub).(mod_name).std_power  = std(abs(Xn).^2);
        result.(sub).(mod_name).snr_mean   = mean(snr_list);

        %% ========================================================
        %  4. CONSTELLATION CHECK METRICS
        %% ========================================================
        result.(sub).(mod_name).mean_abs = mean(abs(Xn));
        result.(sub).(mod_name).std_abs  = std(abs(Xn));
        result.(sub).(mod_name).kurtosis = kurtosis(real(Xn)) + kurtosis(imag(Xn));

    end
end

%% ========== SAVE ==========
save('RX_v3_analysis.mat','result');

disp('RX analysis completed.');
