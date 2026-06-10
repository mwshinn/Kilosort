import logging

import numpy as np
import torch 
from torch.nn.functional import conv1d, max_pool2d, max_pool1d
from tqdm import tqdm

from kilosort import CCG
from kilosort.utils import log_performance

logger = logging.getLogger(__name__)

# Number of learned templates to align per convolution call.
ALIGN_U_CHUNK = 256
# Number of learned templates to score per matching reduction call.
MATCHING_SCORE_CHUNK = 512
# Store the learned-template score matrix on CPU only when it is too large to
# leave enough room for the other per-batch GPU tensors.
MATCHING_B_GPU_TOTAL_FRACTION = 0.75
MATCHING_B_GPU_FREE_FRACTION = 0.85
# Number of detected spikes to update per residual-data matching call.
MATCHING_DATA_CHUNK = 128
# Number of learned templates to update per matching residual call.
MATCHING_TEMPLATE_CHUNK = 2048
# Number of detected spikes to update per learned-template matching call.
MATCHING_CTC_CHUNK = 32
# Number of detected spikes to extract PC features for at once.
MATCHING_FEATURE_CHUNK = 4096


def prepare_extract(xc, yc, U, nC, position_limit, device=torch.device('cuda')):
    """Identify desired channels based on distances and template norms.
    
    Parameters
    ----------
    xc : np.ndarray
        X-coordinates of contact positions on probe.
    yc : np.ndarray
        Y-coordinates of contact positions on probe.
    U : torch.Tensor
        TODO
    nC : int
        Number of nearest channels to use.
    position_limit : float
        Max distance (in microns) between channels that are used to estimate
        spike positions in `postprocessing.compute_spike_positions`.

    Returns
    -------
    iCC : np.ndarray
        For each channel, indices of nC nearest channels.
    iCC_mask : np.ndarray
        For each channel, a 1 if the channel is within 100um and a 0 otherwise.
        Used to control spike position estimate in post-processing.
    iU : torch.Tensor
        For each template, index of channel with greatest norm.
    Ucc : torch.Tensor
        For each template, spatial PC features corresponding to iCC.
    
    """
    ds = (xc - xc[:, np.newaxis])**2 +  (yc - yc[:, np.newaxis])**2 
    iCC = np.argsort(ds, 0)[:nC]
    iCC = torch.from_numpy(iCC).to(device)
    iCC_mask = np.sort(ds, 0)[:nC]
    iCC_mask = iCC_mask < position_limit**2
    iCC_mask = torch.from_numpy(iCC_mask).to(device)
    iU = torch.argmax((U**2).sum(1), -1)
    Ucc = U[torch.arange(U.shape[0]),:,iCC[:,iU]]

    return iCC, iCC_mask, iU, Ucc


def extract(ops, bfile, U, device=torch.device('cuda'), progress_bar=None):
    nC = ops['settings']['nearest_chans']
    position_limit = ops['settings']['position_limit']
    iCC, iCC_mask, iU, Ucc = prepare_extract(
        ops['xc'], ops['yc'], U, nC, position_limit, device=device
        )
    ops['iCC'] = iCC
    ops['iCC_mask'] = iCC_mask
    ops['iU'] = iU
    nt = ops['nt']
    
    tiwave = torch.arange(-(nt//2), nt//2+1, device=device) 
    WtW = prepare_matching(ops, U)
    st = np.zeros((10**6, 3), 'float64')
    tF  = torch.zeros((10**6, nC , ops['settings']['n_pcs']))
    k = 0
    prog = tqdm(
        np.arange(bfile.n_batches, dtype=np.int64),
        miniters=200 if progress_bar else None, 
        mininterval=60 if progress_bar else None
        )
    
    try:
        for ibatch in prog:
            if ibatch % 100 == 0:
                log_performance(logger, 'debug', f'Batch {ibatch}')

            X = bfile.padded_batch_to_torch(ibatch, ops)
            stt, amps, th_amps, Xres = run_matching(ops, X, U, WtW, device=device)
            xfeat = extract_spike_features(
                Xres, iCC, iU, stt, tiwave, ops['wPCA'], Ucc, amps
                )

            if ibatch == 0:
                # Can sometimes get negative spike times for first batch since
                # we're aligning to nt0min, not nt//2, but these should be discarded.
                neg_spikes = (stt[:,0] - nt - nt//2 + ops['nt0min']) < 0
                stt = stt[~neg_spikes,:]
                xfeat = xfeat[:,(~neg_spikes).cpu(),:]
                amps = amps[~neg_spikes,:]
                th_amps = th_amps[~neg_spikes,:]

            nsp = len(stt) 
            if k+nsp>st.shape[0]:                     
                st = np.concatenate((st, np.zeros_like(st)), 0)
                tF  = torch.cat((tF,  torch.zeros_like(tF)), 0)

            t_shift = ibatch * bfile.batch_downsampling * (ops['batch_size'])
            stt = stt.double()
            st[k:k+nsp,0] = ((stt[:,0]-nt) + t_shift).cpu().numpy() - nt//2 + ops['nt0min']
            st[k:k+nsp,1] = stt[:,1].cpu().numpy()
            st[k:k+nsp,2] = th_amps.cpu().numpy().squeeze()
            
            tF[k:k+nsp]  = xfeat.transpose(0,1).cpu()

            k+= nsp
            
            if progress_bar is not None:
                progress_bar.emit(int((ibatch+1) / bfile.n_batches * 100))
    except:
        logger.exception(f'Error in template_matching.extract on batch {ibatch}')
        logger.debug(f'X shape: {X.shape}')
        if 'stt' in locals():
            logger.debug(f'stt shape: {stt.shape}')
        raise

    log_performance(logger, 'debug', f'Batch {ibatch}')

    isort = np.argsort(st[:k,0])
    st = st[isort]
    tF = tF[isort]

    return st, tF, ops


def align_U(U, ops, device=torch.device('cuda')):
    nt = ops['nt']
    padding = nt // 2
    n_templates = U.shape[0]
    wPCA = ops['wPCA'].to(device)
    wTEMP = ops['wTEMP'].to(device).unsqueeze(1)
    Unew = torch.empty((n_templates, wPCA.shape[0], ops['Nchan']),
                       dtype=U.dtype, device='cpu')
    imax = torch.empty(n_templates, dtype=torch.long)

    for start in range(0, n_templates, ALIGN_U_CHUNK):
        end = min(start + ALIGN_U_CHUNK, n_templates)
        context_start = max(0, start - 1)
        context_end = min(n_templates, end + 1)
        Uex = torch.einsum(
            'xyz, zt -> xty', U[context_start:context_end].to(device), wPCA
            )
        X = Uex.reshape(-1, ops['Nchan']).T
        Xconv = conv1d(
            X.unsqueeze(1).contiguous(), wTEMP, padding=padding
            )
        target_start = (start - context_start) * nt
        target_end = target_start + (end - start) * nt
        Xmax = (
            Xconv[..., target_start:target_end]
            .abs().amax((0, 1)).reshape(end - start, nt)
            )
        imax_chunk = torch.argmax(Xmax, 1)

        Uchunk = Uex[start-context_start:end-context_start]
        for j in range(nt):
            ix = imax_chunk == j
            Uchunk[ix] = torch.roll(Uchunk[ix], padding - j, -2)
        Unew[start:end].copy_(torch.einsum('xty, zt -> xzy', Uchunk, wPCA).cpu())
        imax[start:end] = imax_chunk.cpu()

    return Unew.to(device), imax.to(device)


def postprocess_templates(Wall, ops, clu, st, tF, device=torch.device('cuda')):
    Wall2, _ = align_U(Wall, ops, device=device)
    #Wall3, _= remove_duplicates(ops, Wall2)
    Wall3, _, _, _, _ = merging_function(
        ops, Wall2.transpose(1,2), clu, st, tF,
        0.9, 'mu', check_dt=False, device=device
        )
    Wall3 = Wall3.transpose(1,2).to(device)
    return Wall3


def prepare_matching(ops, U):
    nt = ops['nt']
    W = ops['wPCA'].contiguous()
    WtW = conv1d(W.reshape(-1, 1,nt), W.reshape(-1, 1 ,nt), padding = nt) 
    WtW = torch.flip(WtW, [2,])
    return WtW


def use_cpu_matching_scores(n_templates, n_time, dtype, device):
    device = torch.device(device)
    if device.type != 'cuda':
        return False

    b_nbytes = n_templates * n_time * torch.empty((), dtype=dtype).element_size()
    try:
        torch.cuda.empty_cache()
        device_index = device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        free_memory, total_memory = torch.cuda.mem_get_info(device_index)
        return (
            b_nbytes > MATCHING_B_GPU_FREE_FRACTION * free_memory
            or b_nbytes > MATCHING_B_GPU_TOTAL_FRACTION * total_memory
            )
    except Exception:
        return False


def compute_template_scores_cpu(B_features, U):
    n_templates = U.shape[0]
    n_time = B_features.shape[-1]
    B = torch.empty((n_templates, n_time), dtype=B_features.dtype, device='cpu')
    for start in range(0, n_templates, MATCHING_SCORE_CHUNK):
        end = min(start + MATCHING_SCORE_CHUNK, n_templates)
        B[start:end].copy_(
            torch.einsum('ijk, kjl -> il', U[start:end], B_features).cpu()
            )
    return B


def compute_template_scores(B_features, U, device):
    n_templates = U.shape[0]
    n_time = B_features.shape[-1]
    if use_cpu_matching_scores(n_templates, n_time, B_features.dtype, device):
        logger.info('Using CPU-backed learned-template scores for this batch.')
        return compute_template_scores_cpu(B_features, U)

    try:
        return torch.einsum('ijk, kjl -> il', U, B_features)
    except torch.cuda.OutOfMemoryError:
        logger.info(
            'GPU learned-template score allocation failed; falling back to '
            'CPU-backed scores for this batch.'
            )
        torch.cuda.empty_cache()
        return compute_template_scores_cpu(B_features, U)


def get_template_scores(B, iY, iX, device):
    if B.device.type == 'cpu':
        return B[iY.cpu(), iX.cpu()].to(device)
    return B[iY, iX]


def extract_spike_features(Xres, iCC, iU, stt, tiwave, wPCA, Ucc, amps):
    n_spikes = len(stt)
    xfeat = torch.empty(
        (iCC.shape[0], n_spikes, wPCA.shape[0]), dtype=Xres.dtype, device='cpu'
        )

    for start in range(0, n_spikes, MATCHING_FEATURE_CHUNK):
        end = min(start + MATCHING_FEATURE_CHUNK, n_spikes)
        stt_chunk = stt[start:end]
        xfeat_chunk = (
            Xres[iCC[:, iU[stt_chunk[:, 1:2]]], stt_chunk[:, :1] + tiwave]
            @ wPCA.T
            )
        xfeat_chunk += amps[start:end] * Ucc[:, stt_chunk[:, 1]]
        xfeat[:, start:end].copy_(xfeat_chunk.cpu())

    return xfeat


def template_cross_correlations(U, WtW, template_ids, template_start=0,
                                template_end=None):
    if template_end is None:
        template_end = U.shape[0]
    Usel = U[template_ids]
    UtU = torch.einsum('ikl, jml -> ijkm', U[template_start:template_end], Usel)
    return torch.einsum('ijkm, kml -> ijl', UtU, WtW)


def subtract_template_updates(B, U, WtW, iX, iY, amp, trange):
    b_on_cpu = B.device.type == 'cpu'
    for start in range(0, len(iX), MATCHING_CTC_CHUNK):
        end = min(start + MATCHING_CTC_CHUNK, len(iX))
        time_index = iX[start:end] + trange
        if b_on_cpu:
            time_index = time_index.cpu()
        for template_start in range(0, B.shape[0], MATCHING_TEMPLATE_CHUNK):
            template_end = min(template_start + MATCHING_TEMPLATE_CHUNK, B.shape[0])
            ctc = template_cross_correlations(
                U, WtW, iY[start:end, 0], template_start, template_end
                )
            if b_on_cpu:
                b_chunk = B[template_start:template_end, time_index].to(U.device)
                b_chunk -= amp[start:end] * ctc
                B[template_start:template_end, time_index] = b_chunk.cpu()
            else:
                B[template_start:template_end, time_index] -= amp[start:end] * ctc


def subtract_data_updates(Xres, U, W, iX, iY, amp, tiwave):
    for start in range(0, len(iX), MATCHING_DATA_CHUNK):
        end = min(start + MATCHING_DATA_CHUNK, len(iX))
        waveforms = torch.einsum('ijk, jl -> kil', U[iY[start:end, 0]], W)
        Xres[:, iX[start:end] + tiwave] -= amp[start:end] * waveforms


def matching_score_max(B, nm, nt, device=None):
    if device is None:
        device = B.device
    device = torch.device(device)
    Cfmax = torch.zeros(B.shape[1], dtype=B.dtype, device=device)
    imax = torch.zeros(B.shape[1], dtype=torch.long, device=device)

    for start in range(0, B.shape[0], MATCHING_SCORE_CHUNK):
        end = min(start + MATCHING_SCORE_CHUNK, B.shape[0])
        if B.device.type == 'cpu':
            Cf = B[start:end].to(device)
            Cf.clamp_min_(0).square_()
        else:
            Cf = torch.relu(B[start:end]).square_()
        Cf /= nm[start:end].to(device).unsqueeze(-1)
        Cf[:, :nt] = 0
        Cf[:, -nt:] = 0
        Cfmax_chunk, imax_chunk = torch.max(Cf, 0)
        is_better = Cfmax_chunk > Cfmax
        Cfmax[is_better] = Cfmax_chunk[is_better]
        imax[is_better] = imax_chunk[is_better] + start

    return Cfmax, imax


def run_matching(ops, X, U, WtW, device=torch.device('cuda')):
    Th = ops['Th_learned']
    nt = ops['nt']
    max_peels = ops['max_peels']
    W = ops['wPCA'].contiguous()

    nm = (U**2).sum(-1).sum(-1)
    #mu = nm**.5 
    #U2 = U / mu.unsqueeze(-1).unsqueeze(-1)

    B_features = conv1d(X.unsqueeze(1), W.unsqueeze(1), padding=nt//2)
    B = compute_template_scores(B_features, U, device)
    del B_features

    trange = torch.arange(-nt, nt+1, device=device) 
    tiwave = torch.arange(-(nt//2), nt//2+1, device=device) 

    st = torch.zeros((100000,2), dtype = torch.int64, device = device)
    amps = torch.zeros((100000,1), dtype = torch.float, device = device)
    th_amps = torch.zeros((100000,1), dtype = torch.float, device = device)
    k = 0

    Xres = X.clone()
    lam = 20

    for t in range(max_peels):
        # Cf = 2 * B - nm.unsqueeze(-1) 
        # Cf is shape (n_units, n_times)
        Cfmax, imax = matching_score_max(B, nm, nt, device=device)
        #a = 1 + lam
        #b = torch.relu(B) + lam * mu.unsqueeze(-1)
        #Cf = b**2 / a - lam * mu.unsqueeze(-1)**2

        Cmax  = max_pool1d(Cfmax.unsqueeze(0).unsqueeze(0), (2*nt+1), stride=1, padding=(nt))

        #print(Cfmax.shape)
        #import pdb; pdb.set_trace()
        cnd1 = Cmax[0,0] > Th**2
        cnd2 = torch.abs(Cmax[0,0] - Cfmax) < 1e-9
        xs = torch.nonzero(cnd1 * cnd2)

        
        if len(xs)==0:
            #print('iter %d'%t)
            break

        iX = xs[:,:1]
        iY = imax[iX]

        #isort = torch.sort(iX)

        nsp = len(iX)
        st[k:k+nsp, 0] = iX[:,0]
        st[k:k+nsp, 1] = iY[:,0]
        amps[k:k+nsp] = get_template_scores(B, iY, iX, device) / nm[iY]
        amp = amps[k:k+nsp]
        th_amps[k:k+nsp] = Cmax[0, 0, iX[:,0], None]**.5

        k+= nsp

        #amp = B[iY,iX] 

        n = 2
        for j in range(n):
            subtract_data_updates(
                Xres, U, W, iX[j::n], iY[j::n], amp[j::n], tiwave
                )
            subtract_template_updates(
                B, U, WtW, iX[j::n], iY[j::n], amp[j::n], trange
                )

    st = st[:k]
    amps = amps[:k]
    th_amps = th_amps[:k]

    return  st, amps, th_amps, Xres


def merging_function(ops, Wall, clu, st, tF, r_thresh=0.5, mode='ccg', check_dt=True,
                     device=torch.device('cuda')):
    clu2 = clu.copy()
    clu_unq, ns = np.unique(clu2, return_counts = True)

    Ww = Wall.to(device)
    NN = len(Ww)

    isort = np.argsort(ns)[::-1]

    is_merged = np.zeros(NN, 'bool')
    is_good = np.zeros(NN,)

    acg_threshold = ops['settings']['acg_threshold']
    ccg_threshold = ops['settings']['ccg_threshold']
    if mode == 'ccg':
        is_ref, est_contam_rate = CCG.refract(clu, st[:,0]/ops['fs'],
                                              acg_threshold=acg_threshold,
                                              ccg_threshold=ccg_threshold)

    nt = ops['nt']
    W = ops['wPCA'].contiguous()
    WtW = conv1d(W.reshape(-1, 1,nt), W.reshape(-1, 1 ,nt), padding = nt) 
    WtW = torch.flip(WtW, [2,])

    t = 0
    nmerge = 0
    while t<NN:
        #if t%100==0:
            #print(t, nmerge)

        kk = clu_unq[isort[t]]

        if (mode == 'ccg') and is_ref[kk]==0:
            t += 1
            continue

        if is_merged[kk]:            
            t += 1
            continue

        mu = (Ww**2).sum((1,2), keepdims=True)**.5
        Wnorm = Ww / (1e-6 + mu)

        UtU = torch.einsum('lk, jlm -> jkm',  Wnorm[kk], Wnorm)
        ctc = torch.einsum('jkm, kml -> jl', UtU, WtW)

        cmax, imax = ctc.max(1)
        cmax[kk] = 0

        jsort = np.argsort(cmax.cpu().numpy())[::-1]

        if mode == 'ccg':
            st0 = st[:,0][clu2==kk] / ops['fs']
        
        is_ccg  = 0
        for j in range(NN):
            jj = jsort[j]
            if cmax[jj] < r_thresh:
                break
            # compare with CCG
            if mode == 'ccg':
                st1 = st[:,0][clu2==jj] / ops['fs']
                _, is_ccg, _ = CCG.check_CCG(st0, st1, acg_threshold=acg_threshold,
                                             ccg_threshold=ccg_threshold)        
            else:
                dmu = 2 * (mu[kk] - mu[jj]) / (mu[kk] + mu[jj])
                is_ccg = dmu.abs() < 0.2

            if is_ccg:
                is_merged[jj] = 1
                dt = (imax[kk] -imax[jj]).item()
                if dt != 0 and check_dt:
                    # Get spike indices for cluster jj
                    idx = (clu2 == jj)
                    # Update tF and Wall with shifted features
                    tF, Wall = roll_features(W, tF, Ww, idx, jj, dt)
                    # Shift spike times
                    st[idx,0] -= dt
                
                Ww[kk] = ns[kk]/(ns[kk]+ns[jj]) * Ww[kk] + ns[jj]/(ns[kk]+ns[jj]) * Ww[jj]            
                Ww[jj] = 0
                ns[kk] += ns[jj]
                ns[jj] = 0
                clu2[clu2==jj] = kk            

                break

        if is_ccg==0:            
            t +=1    
        else:                
            nmerge+=1
    
    imap = np.cumsum((~is_merged).astype('int32')) - 1
    if imap.size > 0:
        # Otherwise, everything has been merged into a single cluster
        clu2 = imap[clu2]

    Ww = Ww[~is_merged]

    if mode == 'ccg':
        is_ref = is_ref[~is_merged]
    else:
        is_ref = None

    sorted_idx = np.argsort(st[:,0])
    st = np.take_along_axis(st, sorted_idx[..., np.newaxis], axis=0)
    clu2 = clu2[sorted_idx]
    tensor_idx = torch.from_numpy(sorted_idx)
    tF = tF[tensor_idx]

    return Ww.cpu(), clu2, is_ref, st, tF


def roll_features(wPCA, tF, Wall, spike_idx, clust_idx, dt):
    W = wPCA.cpu()
    # Project from PC space back to sample time, shift by dt
    feats = torch.roll(tF[spike_idx] @ W, shifts=dt, dims=2)
    temps = torch.roll(Wall[clust_idx:clust_idx+1] @ wPCA, shifts=dt, dims=2)

    # For values that "rolled over the edge," set equal to next closest bin
    if dt > 0:
        feats[:,:,:dt] = feats[:,:,dt].unsqueeze(-1)
        temps[:,:,:dt] = temps[:,:,dt].unsqueeze(-1)
    elif dt < 0:
        feats[:,:,dt:] = feats[:,:,dt-1].unsqueeze(-1)
        temps[:,:,dt:] = temps[:,:,dt-1].unsqueeze(-1)

    # Project back to PC space and update tF
    tF[spike_idx] = feats @ W.T
    Wall[clust_idx] = temps @ wPCA.T

    return tF, Wall
