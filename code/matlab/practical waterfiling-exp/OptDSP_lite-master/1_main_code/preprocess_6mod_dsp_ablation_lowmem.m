%% preprocess_6mod_dsp_ablation_lowmem.m
% 6调制 DSP依赖性消融预处理：NoEQ / Blind-like
% noeq      = LTS同步 + LTS-CFO + FFT，不做LTS信道估计/均衡
% blindlike = LTS同步仅定位 + CP-CFO + FFT，不做LTS信道估计/均衡
% 输出目录：2_Data_Results/preprocessed_uniform_qam_rx_dsp_ablation/2026.06.28/<mode>

clear; clear global; close all; clc;
project_root = 'D:\Project_code\OptDSP_FSO_2\practical waterfiling-exp\OptDSP_lite-master';
data_root    = fullfile(project_root, '2_Data_Results');
addpath(genpath(project_root));
global PROG; PROG.showMessagesLevel = 0; try, initProg(); catch, end

cfg = struct();
cfg.project_root = project_root; cfg.data_root = data_root; cfg.date_tag = '2026.06.28';
cfg.mod_list = {'QPSK','16QAM','32QAM','64QAM','128QAM','256QAM'};
cfg.mod_label_list = [0 1 2 3 4 5];
cfg.turb_subdirs = {'sub01','sub03'}; cfg.turb_names = {'weak','strong'}; cfg.turb_labels = [0 1];
cfg.modes = {'noeq','blindlike'};   % 如需重做均衡版本，可改为 {'eq','noeq','blindlike'}
cfg.out_root_all = fullfile(data_root, 'preprocessed_uniform_qam_rx_dsp_ablation', cfg.date_tag);
if ~exist(cfg.out_root_all,'dir'), mkdir(cfg.out_root_all); end

cfg.Fs_rx = 80e9; cfg.Fs_base = 16e9; cfg.n_frames_per_file = 3;
cfg.zeros_head = 80; cfg.n_fft = 256; cfg.n_guard = 16; cfg.n_syms = 128;
cfg.carrier_loc = 4:126; cfg.n_sc = length(cfg.carrier_loc);
cfg.sym_len = cfg.n_fft + cfg.n_guard;
cfg.frame_len_16 = cfg.zeros_head + cfg.n_guard + 2*cfg.n_fft + cfg.sym_len*cfg.n_syms;
cfg.frame_pre_lts = cfg.zeros_head + cfg.n_guard - 5; cfg.next_search_backoff = 800;
cfg.cdm_bins = 64; cfg.cdm_clip = 3.0; cfg.debug_max_jobs = inf; cfg.max_files_per_turb_per_mod = inf;
cfg.continue_on_frame_error = true; cfg.progress_every_n_files = 5;

LTS = make_lts_local(cfg.n_fft);
jobs = discover_jobs(cfg); if isempty(jobs), error('No RX bin files found.'); end
if isfinite(cfg.debug_max_jobs), jobs = jobs(1:min(numel(jobs),cfg.debug_max_jobs)); end

fprintf('\nDSP ablation preprocessing | jobs=%d | modes=%s\n', numel(jobs), strjoin(cfg.modes,','));

fids = containers.Map(); frame_ok = containers.Map(); frame_fail = containers.Map(); gid = containers.Map();
for m = 1:numel(cfg.modes)
    mode = cfg.modes{m}; root = fullfile(cfg.out_root_all,mode); if ~exist(root,'dir'), mkdir(root); end
    fid = fopen(fullfile(root,'manifest_all.csv'),'w'); write_manifest_header(fid); fids(mode)=fid;
    frame_ok(mode)=0; frame_fail(mode)=0; gid(mode)=0;
end

t0 = tic;
for j = 1:numel(jobs)
    job = jobs(j);
    fprintf('\n[%d/%d] %s | %s | %s\n', j, numel(jobs), job.mod_name, job.turb_name, job.rx_name);
    try
        [rx80,~] = read_keysight_bin_robust_real_local(job.rx_file);
        rx80 = rx80(:).'; rx80 = rx80 - mean(rx80); rx80 = rx80 ./ (rms(rx80)+eps);
        rx16 = resample(rx80, cfg.Fs_base, cfg.Fs_rx); clear rx80;
        rx16 = rx16(:).'; rx16 = rx16 - mean(rx16); rx16 = rx16 ./ (mean(abs(rx16))+eps);
        wrap_len = min(length(rx16), 3*cfg.frame_len_16); rx = [rx16, rx16(1:wrap_len)]; clear rx16;
        cursor = 1;
        for rk = 1:cfg.n_frames_per_file
            try
                search_sig = rx(cursor:end);
                [lts_rel, frame_rel] = find_frame_lts(search_sig, cfg); clear search_sig;
                lts_abs = cursor + lts_rel - 1; frame_abs = cursor + frame_rel - 1;
                dem = demod_modes(rx, lts_abs, LTS, cfg);
                for mi = 1:numel(cfg.modes)
                    mode = cfg.modes{mi};
                    switch mode
                        case 'eq',        rx_sc = dem.rx_sc_eq;
                        case 'noeq',      rx_sc = dem.rx_sc_noeq;
                        case 'blindlike', rx_sc = dem.rx_sc_blindlike;
                    end
                    [mat_path, out_dir] = out_path(cfg, mode, job, rk); if ~exist(out_dir,'dir'), mkdir(out_dir); end
                    cdm64 = make_cdm(rx_sc, cfg.cdm_bins, cfg.cdm_clip);
                    [blind_stats, blind_stats_names] = make_stats(rx_sc);
                    sample = struct(); sample.rx_sc = single(rx_sc); sample.cdm64 = single(cdm64);
                    sample.blind_stats = single(blind_stats(:).'); sample.blind_stats_names = blind_stats_names;
                    sample.mod_name = job.mod_name; sample.mod_label = int32(job.mod_label);
                    sample.turb_name = job.turb_name; sample.turb_label = int32(job.turb_label); sample.turb_subdir = job.turb_subdir;
                    sample.rx_file = job.rx_file; sample.rx_name = job.rx_name; sample.sig_idx = int32(job.sig_idx); sample.rx_frame_idx = int32(rk);
                    sample.lts_start_abs = int64(lts_abs); sample.frame_start_abs = int64(frame_abs); sample.dsp_mode = mode;
                    sample.cfo_lts = single(dem.cfo_lts); sample.cfo_cp = single(dem.cfo_cp); sample.n_use = int32(dem.n_use);
                    sample.snr_sc_db = single(NaN(cfg.n_sc,1)); sample.snr_frame_db = single(NaN);
                    save(mat_path, '-struct', 'sample', '-v7');
                    gid(mode)=gid(mode)+1; frame_ok(mode)=frame_ok(mode)+1;
                    rec = make_rec(job,rk,mat_path,'ok','',lts_abs,frame_abs,dem.cfo_lts,dem.cfo_cp,dem.n_use,blind_stats,mode,gid(mode));
                    write_manifest_row(fids(mode), rec);
                end
                cursor_next = frame_abs + cfg.frame_len_16 - cfg.next_search_backoff;
                if cursor_next <= cursor, cursor_next = cursor + round(0.8*cfg.frame_len_16); end
                cursor = cursor_next;
            catch MEf
                fprintf('  frame %d failed: %s\n', rk, MEf.message);
                for mi=1:numel(cfg.modes)
                    mode=cfg.modes{mi}; frame_fail(mode)=frame_fail(mode)+1;
                    rec = make_rec(job,rk,'','failed',MEf.message,NaN,NaN,NaN,NaN,NaN,NaN(1,16),mode,NaN);
                    write_manifest_row(fids(mode), rec);
                end
                cursor = cursor + round(0.8*cfg.frame_len_16);
            end
        end
        clear rx;
    catch ME
        fprintf('  file failed: %s\n', ME.message);
        for mi=1:numel(cfg.modes)
            mode=cfg.modes{mi}; frame_fail(mode)=frame_fail(mode)+cfg.n_frames_per_file;
            rec = make_rec(job,NaN,'','failed_file',ME.message,NaN,NaN,NaN,NaN,NaN,NaN(1,16),mode,NaN);
            write_manifest_row(fids(mode), rec);
        end
    end
    if mod(j,cfg.progress_every_n_files)==0 || j==numel(jobs)
        el=toc(t0); rem=el/j*(numel(jobs)-j); fprintf('[progress] %.1f min elapsed, %.1f min remaining\n', el/60, rem/60);
    end
end

for mi=1:numel(cfg.modes)
    mode=cfg.modes{mi}; fclose(fids(mode));
    fid=fopen(fullfile(cfg.out_root_all,mode,'run_summary_dsp_ablation.txt'),'w');
    fprintf(fid,'mode=%s\nprocessed_frames=%d\nfailed_frames=%d\ntime_min=%.3f\n',mode,frame_ok(mode),frame_fail(mode),toc(t0)/60); fclose(fid);
end
fprintf('\nDone. Total %.1f min\n', toc(t0)/60);

%% ===================== functions =====================
function jobs = discover_jobs(cfg)
    jobs=struct([]); id=0;
    for mi=1:numel(cfg.mod_list)
        modn=cfg.mod_list{mi}; lab=cfg.mod_label_list(mi);
        for ti=1:numel(cfg.turb_subdirs)
            rx_dir=fullfile(cfg.data_root,'rx_data',cfg.date_tag,modn,cfg.turb_subdirs{ti});
            if ~exist(rx_dir,'dir'), warning('missing %s',rx_dir); continue; end
            fs=dir(fullfile(rx_dir,'*.bin')); fs=sort_files(fs);
            if isfinite(cfg.max_files_per_turb_per_mod), fs=fs(1:min(numel(fs),cfg.max_files_per_turb_per_mod)); end
            fprintf('[discover] %s %s %d files\n',modn,cfg.turb_names{ti},numel(fs));
            for k=1:numel(fs)
                id=id+1; jobs(id).mod_name=modn; jobs(id).mod_label=lab; jobs(id).turb_subdir=cfg.turb_subdirs{ti};
                jobs(id).turb_name=cfg.turb_names{ti}; jobs(id).turb_label=cfg.turb_labels(ti); jobs(id).rx_file=fullfile(fs(k).folder,fs(k).name);
                jobs(id).rx_name=fs(k).name; jobs(id).sig_idx=infer_idx(fs(k).name,k);
            end
        end
    end
end
function [mat_path,out_dir]=out_path(cfg,mode,job,rk)
    file_base=erase(job.rx_name,'.bin'); safe=regexprep(file_base,'[^\w\-]','_');
    out_dir=fullfile(cfg.out_root_all,mode,job.mod_name,job.turb_name,sprintf('%s_sig%04d_%s',job.mod_name,job.sig_idx,safe));
    mat_path=fullfile(out_dir,sprintf('frame_%02d.mat',rk));
end
function [lts_start, frame_start]=find_frame_lts(rx,cfg)
    n_fft=cfg.n_fft; n_guard=cfg.n_guard; symbits=cfg.zeros_head+n_guard+2*n_fft+(n_fft+n_guard)*cfg.n_syms;
    search_len=min(length(rx),2*symbits); if search_len<symbits, error('input too short'); end
    [detected_packet,edge_index]=packet_edge_power_dect(rx(1:search_len),cfg.zeros_head);
    load('LongTrainSym_ini.mat','LongTrainSym_ini'); LTS_f=LongTrainSym_ini(1:n_fft); LTS_f([1 n_fft/2+1])=0;
    ltrs=LTS_f; ltrs(1,n_fft/2+2:n_fft)=conj(ltrs(1,n_fft/2:-1:2));
    [fine_time_est,~,~]=rx_fine_time_sync_cross_corr(detected_packet,n_guard,ltrs,cfg.zeros_head,0);
    lts_start=edge_index+fine_time_est-1; frame_start=max(1,lts_start-cfg.frame_pre_lts);
end
function dem=demod_modes(rx,lts_start,LTS,cfg)
    n_fft=cfg.n_fft; n_guard=cfg.n_guard; sym_len=cfg.sym_len; rem=length(rx)-lts_start+1;
    lts1=rx(lts_start:lts_start+n_fft-1); lts2=rx(lts_start+n_fft:lts_start+2*n_fft-1);
    cfo_lts=angle(sum(lts1(:).*conj(lts2(:))))/(2*pi*n_fft); n=0:rem-1;
    rx_lts=rx(lts_start:end).*exp(-1j*2*pi*cfo_lts*n/n_fft);
    dp_lts=rx_lts(2*n_fft+1:end); nd=floor(length(dp_lts)/sym_len); n_use=min(nd,cfg.n_syms); if n_use<=0,error('no symbols');end
    fd_noeq=fft_payload(dp_lts,n_fft,n_guard,n_use); rx_sc_noeq=fd_noeq(cfg.carrier_loc,:);
    lts_avg=(rx_lts(1:n_fft).'+rx_lts(n_fft+1:2*n_fft).')/2; lts_fd=fft(lts_avg,n_fft)/sqrt(n_fft);
    H=lts_fd./(LTS.freq(:)+1e-12); H(abs(LTS.freq(:))<0.5)=1; rx_sc_eq=(fd_noeq./H); rx_sc_eq=rx_sc_eq(cfg.carrier_loc,:);
    raw=rx(lts_start:end); cfo_cp=est_cfo_cp(raw,n_fft,n_guard,n_use); rx_cp=raw.*exp(-1j*2*pi*cfo_cp*n/n_fft);
    fd_blind=fft_payload(rx_cp(2*n_fft+1:end),n_fft,n_guard,n_use); rx_sc_blind=fd_blind(cfg.carrier_loc,:);
    dem=struct('rx_sc_eq',rx_sc_eq,'rx_sc_noeq',rx_sc_noeq,'rx_sc_blindlike',rx_sc_blind,'cfo_lts',cfo_lts,'cfo_cp',cfo_cp,'n_use',n_use);
end
function fd=fft_payload(dp_all,n_fft,n_guard,n_use)
    sym_len=n_fft+n_guard; dp=dp_all(1:n_use*sym_len); dm=reshape(dp,sym_len,n_use); dn=dm(n_guard+1:end,:); fd=fft(dn,n_fft,1)/sqrt(n_fft);
end
function cfo=est_cfo_cp(raw,n_fft,n_guard,n_use)
    sym_len=n_fft+n_guard; dp=raw(2*n_fft+1:end); nd=floor(length(dp)/sym_len); n_use=min([n_use,nd,32]); acc=0;
    for k=1:n_use
        st=(k-1)*sym_len+1; sym=dp(st:st+sym_len-1); cp=sym(1:n_guard); tail=sym(n_fft+1:n_fft+n_guard); acc=acc+sum(conj(cp(:)).*tail(:));
    end
    if abs(acc)<eps, cfo=0; else, cfo=angle(acc)/(2*pi); end
end
function cdm=make_cdm(rx_sc,nbin,clipv)
    z=rx_sc(:); z=z(isfinite(real(z))&isfinite(imag(z))); if isempty(z), cdm=zeros(nbin,nbin,'single'); return; end
    z=z-mean(z); z=z./sqrt(mean(abs(z).^2)+eps); zr=max(min(real(z),clipv),-clipv); zi=max(min(imag(z),clipv),-clipv);
    edges=linspace(-clipv,clipv,nbin+1); H=histcounts2(zi,zr,edges,edges); cdm=log1p(H); cdm=single(cdm./(max(cdm(:))+eps));
end
function [s,names]=make_stats(rx_sc)
    z=rx_sc(:); z=z(isfinite(real(z))&isfinite(imag(z))); names={'amp_mean','amp_std','amp_skew','amp_kurt','papr_db','i_mean','i_std','i_skew','i_kurt','q_mean','q_std','q_skew','q_kurt','phase_diff_std','phase_concentration','iq_corr'};
    if isempty(z), s=NaN(1,16); return; end
    z=z-mean(z); z=z./sqrt(mean(abs(z).^2)+eps); a=abs(z); ii=real(z); qq=imag(z); dph=diff(unwrap(angle(z)));
    if std(ii)<eps||std(qq)<eps, iqc=0; else, C=corrcoef(ii,qq); iqc=C(1,2); end
    s=[mean(a),std(a),skew_m(a),kurt_m(a),10*log10(max(abs(z).^2)/(mean(abs(z).^2)+eps)),mean(ii),std(ii),skew_m(ii),kurt_m(ii),mean(qq),std(qq),skew_m(qq),kurt_m(qq),std(dph),abs(mean(exp(1j*angle(z)))),iqc];
end
function y=skew_m(x), x=x(:); x=x(isfinite(x)); if numel(x)<3,y=NaN;else,y=mean(((x-mean(x))./(std(x)+eps)).^3);end,end
function y=kurt_m(x), x=x(:); x=x(isfinite(x)); if numel(x)<4,y=NaN;else,y=mean(((x-mean(x))./(std(x)+eps)).^4);end,end
function rec=make_rec(job,rk,out,status,msg,lts,frame,cfo_lts,cfo_cp,n_use,bs,mode,gid)
    if isempty(bs)||all(isnan(bs(:))), bs=NaN(1,16); end; bs=double(bs(:).'); if numel(bs)<16, bs=[bs,NaN(1,16-numel(bs))]; end
    rec=struct('global_frame_id',gid,'status',status,'message',clean(msg),'mod_name',job.mod_name,'mod_label',job.mod_label,'turb_name',job.turb_name,'turb_label',job.turb_label,'turb_subdir',job.turb_subdir,'rx_file',job.rx_file,'rx_name',job.rx_name,'sig_idx',job.sig_idx,'rx_frame_idx',rk,'lts_start_abs',lts,'frame_start_abs',frame,'cfo_lts',cfo_lts,'cfo_cp',cfo_cp,'n_use',n_use,'dsp_mode',mode,'blind_stats',bs(1:16),'out_mat',out);
end
function write_manifest_header(fid)
    fprintf(fid,'global_frame_id,status,message,mod_name,mod_label,turb_name,turb_label,turb_subdir,rx_file,rx_name,sig_idx,rx_frame_idx,lts_start_abs,frame_start_abs,cfo_lts,cfo_cp,n_use,dsp_mode,amp_mean,amp_std,amp_skew,amp_kurt,papr_db,i_mean,i_std,i_skew,i_kurt,q_mean,q_std,q_skew,q_kurt,phase_diff_std,phase_concentration,iq_corr,out_mat\n');
end
function write_manifest_row(fid,r)
    b=r.blind_stats; fprintf(fid,'%g,%s,%s,%s,%d,%s,%d,%s,%s,%s,%d,%g,%g,%g,%.12g,%.12g,%g,%s,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%s\n',r.global_frame_id,r.status,clean(r.message),r.mod_name,r.mod_label,r.turb_name,r.turb_label,r.turb_subdir,clean(r.rx_file),clean(r.rx_name),r.sig_idx,r.rx_frame_idx,r.lts_start_abs,r.frame_start_abs,r.cfo_lts,r.cfo_cp,r.n_use,r.dsp_mode,b(1),b(2),b(3),b(4),b(5),b(6),b(7),b(8),b(9),b(10),b(11),b(12),b(13),b(14),b(15),b(16),clean(r.out_mat));
end
function t=clean(t), if isempty(t),t='';return;end, if isstring(t),t=char(t);end, if isnumeric(t),t=num2str(t);end, t=strrep(t,',',';'); t=strrep(t,newline,' '); end
function LTS=make_lts_local(n_fft), load('LongTrainSym_ini.mat','LongTrainSym_ini'); f=LongTrainSym_ini(1:n_fft); f([1 n_fft/2+1])=0; f(1,n_fft/2+2:n_fft)=conj(f(1,n_fft/2:-1:2)); LTS.freq=f(:); end
function fs=sort_files(fs), nums=zeros(numel(fs),1); for i=1:numel(fs),nums(i)=infer_idx(fs(i).name,i);end,[~,idx]=sortrows([nums(:),(1:numel(fs)).']);fs=fs(idx); end
function idx=infer_idx(name,fb), tok=regexp(name,'(\d+)','tokens','once'); if isempty(tok),idx=fb;else,idx=str2double(tok{1});if isnan(idx),idx=fb;end,end,end
function [y,info]=read_keysight_bin_robust_real_local(filename)
    info=struct(); try,[y,info]=read_keysight_real(filename,false);return;catch ME1,info.e1=ME1.message;end; try,[y,info]=read_keysight_real(filename,true);return;catch ME2,error('read failed: %s | %s',info.e1,ME2.message);end
end
function [y,info]=read_keysight_real(filename,force_infer)
    fid=fopen(filename,'rb','ieee-le'); if fid==-1,error('open fail');end; c=onCleanup(@()fclose(fid)); %#ok<NASGU>
    fread(fid,2,'*char'); fread(fid,2,'*char'); fread(fid,5,'int32'); num_points=fread(fid,1,'int32'); fread(fid,1,'int32'); fread(fid,1,'float32'); fread(fid,3,'float64'); fread(fid,2,'int32'); fread(fid,16,'*char'); fread(fid,16,'*char'); fread(fid,24,'*char'); fread(fid,16,'*char'); fread(fid,1,'float64'); fread(fid,1,'uint32'); fread(fid,1,'int32'); fread(fid,1,'int16'); bpp=fread(fid,1,'int16'); buf=fread(fid,1,'int32'); data_start=ftell(fid); d=dir(filename); rem=d.bytes-data_start;
    if ~force_infer, cand=double(bpp); else, cand=unique([double(bpp),round(double(buf)/double(num_points)),round(double(rem)/double(num_points)),4,2,1,8],'stable'); cand=cand(ismember(cand,[1 2 4 8])); end
    last=''; for k=1:numel(cand), try, fseek(fid,data_start,'bof'); raw=read_raw(fid,double(num_points),cand(k)); raw=double(raw(:)).'; if length(raw)<1000||std(raw(1:min(5000,end)))==0,error('bad raw');end; y=raw; info=struct('bpp',bpp,'inferred_bpp',cand(k)); return; catch ME,last=ME.message; end,end; error(last);
end
function raw=read_raw(fid,n,bpp), switch bpp, case 4,raw=fread(fid,n,'float32'); case 2,raw=fread(fid,n,'int16'); case 1,raw=fread(fid,n,'int8'); case 8,raw=fread(fid,n,'double'); otherwise,error('bad bpp'); end,end
