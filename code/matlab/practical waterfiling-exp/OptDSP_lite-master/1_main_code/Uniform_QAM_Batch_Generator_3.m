
%% ============================================================
%  Uniform QAM Batch Generator v3
%  每种调制 50 bins × 3 frames × 2 turbulence folders
% ============================================================

clear; clc;

%% ========== CONFIG ==========
cfg.mods = {'QPSK','16QAM','32QAM','64QAM','128QAM','256QAM'};
cfg.M = [4,16,32,64,128,256];

cfg.n_bin = 50;            % 每种调制每个sub文件夹50个bin
cfg.n_frame = 3;           % 每个bin 3帧

cfg.subfolders = {'sub01','sub03'};   % 弱/强湍流

cfg.root_dir = './TX_UNIFORM_BATCH_V3/';

cfg.fs_scale = 1/sqrt(512); % OFDM功率归一化（保持一致）

rng(2026);

%% ========== LOOP ==========
for s = 1:length(cfg.subfolders)

    sub = cfg.subfolders{s};

    for m = 1:length(cfg.mods)

        mod_name = cfg.mods{m};
        M = cfg.M(m);

        % ---------- create constellation ----------
        c = generate_constellation(mod_name);

        % normalize (unit avg power)
        c = c / sqrt(mean(abs(c).^2));

        out_dir = fullfile(cfg.root_dir, sub, mod_name);
        if ~exist(out_dir, 'dir')
            mkdir(out_dir);
        end

        fprintf('\n[%s - %s] generating...\n', sub, mod_name);

        for b = 1:cfg.n_bin

            tx_all = [];

            for f = 1:cfg.n_frame

                %% ========== generate random QAM symbols ==========
                n_sc = 64;
                n_sym = 32;

                idx = randi([1 M], n_sc, n_sym);
                X = c(idx);

                %% ========== OFDM ==========
                x_time = ifft(X, [], 1);
                x_cp = [x_time(end-15:end,:); x_time];

                tx = x_cp(:);

                %% ========== power normalize ==========
                tx = cfg.fs_scale * tx;

                %% ========== append 3 frames ==========
                tx_all = [tx_all; tx];

            end

            %% ========== save BIN ==========
            fname = fullfile(out_dir, sprintf('sig_%04d.bin', b));

            fid = fopen(fname, 'wb');
            fwrite(fid, [real(tx_all), imag(tx_all)].', 'float32');
            fclose(fid);

            fprintf('Saved %s\n', fname);
        end
    end
end


%% ============================================================
function c = generate_constellation(name)

switch name

    case 'QPSK'
        c = [1+1j, 1-1j, -1+1j, -1-1j];

    case '16QAM'
        lv = [-3 -1 1 3];
        [I,Q] = meshgrid(lv, lv);
        c = I(:) + 1j*Q(:);

    case '32QAM'
        lv = [-5 -3 -1 1 3 5];
        [I,Q] = meshgrid(lv, lv);
        mask = ~((abs(I)==5 & abs(Q)==5));
        c = (I(mask) + 1j*Q(mask));

    case '64QAM'
        lv = [-7 -5 -3 -1 1 3 5 7];
        [I,Q] = meshgrid(lv, lv);
        c = I(:) + 1j*Q(:);

    case '128QAM'
        lv = [-11 -9 -7 -5 -3 -1 1 3 5 7 9 11];
        [I,Q] = meshgrid(lv, lv);
        mask = ~((abs(I)>=9 & abs(Q)>=9));
        c = (I(mask) + 1j*Q(mask));

    case '256QAM'
        lv = [-15 -13 -11 -9 -7 -5 -3 -1 1 3 5 7 9 11 13 15];
        [I,Q] = meshgrid(lv, lv);
        c = I(:) + 1j*Q(:);

end

end

