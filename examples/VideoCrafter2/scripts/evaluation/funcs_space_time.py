import os, sys, glob
import numpy as np
from collections import OrderedDict
from decord import VideoReader, cpu
import cv2
import torch.fft as fft

import torch
import torchvision
sys.path.insert(1, os.path.join(sys.path[0], '..', '..'))
from lvdm.models.samplers.ddim import DDIMSampler


def snr_calculation(signal_band_power, noise_band_power):
    # Convert inputs to float tensors if they are not already tensors
    if not isinstance(signal_band_power, torch.Tensor):
        signal_band_power = torch.tensor(signal_band_power, dtype=torch.float32)
    if not isinstance(noise_band_power, torch.Tensor):
        noise_band_power = torch.tensor(noise_band_power, dtype=torch.float32)

    # Calculate SNR in dB
    snr = 10 * torch.log10(signal_band_power / noise_band_power)
    return snr

def analyze_frequency_components_torch(feature_map):
    # Assuming feature_map has shape (1, C, T, H, W)
    feature_map = feature_map.squeeze(0)
    C, T, H, W = feature_map.shape

    # Separate FFT operations for time and spatial dimensions
    feature_map_freq_time = fft.fftn(feature_map, dim=(-3,))
    feature_map_freq_space = fft.fftn(feature_map, dim=(-2, -1))
    feature_map_freq = fft.fftn(feature_map, dim=(-3, -2, -1))

    # Shift to center the zero frequencies
    feature_map_freq_time = fft.fftshift(feature_map_freq_time, dim=(-3,))
    feature_map_freq_space = fft.fftshift(feature_map_freq_space, dim=(-2, -1))
    feature_map_freq = fft.fftshift(feature_map_freq, dim=(-3, -2, -1))

    return torch.abs(feature_map_freq), torch.abs(feature_map_freq_space), torch.abs(feature_map_freq_time)


def identify_high_low_masks(mag, mag_space, mag_time):

    def generate_mask(threshold, C, T, H, W):
        t_range = torch.zeros(T)
        h_range = torch.zeros(H)
        w_range = torch.zeros(W)

        t_range[int(0.0 * T):int(threshold * T)] = 1
        t_range[int((1 - threshold) * T):int(1.0 * T)] = 1

        h_range[int(0.0 * H):int(threshold * H)] = 1
        h_range[int((1 - threshold) * H):int(1.0 * H)] = 1

        w_range[int(0.0 * W):int(threshold * W)] = 1
        w_range[int((1 - threshold) * W):int(1.0 * W)] = 1

        mask = t_range[:, None, None] + h_range[None, :, None] + w_range[None, None, :]
        mask[mask>0] = 1

        mask_time = t_range[:, None, None].repeat(1, H, W)

        mask_space = (h_range[:, None] + w_range[None, :]).unsqueeze(0).repeat(T, 1, 1)
        mask_space[mask_space>0] = 1

        return mask.unsqueeze(0).repeat(C, 1, 1, 1), mask_time.unsqueeze(0).repeat(C, 1, 1, 1), mask_space.unsqueeze(0).repeat(C, 1, 1, 1)
        

    # Assuming mag is the magnitude of the spatial FFT and has shape (C, T, H, W)
    C, T, H, W = mag.shape
    high_thresh = 0.125
    mid_high_thresh = 0.25
    mid_low_thresh = 0.375
    low_thresh = 0.5

    high_mask, high_mask_time, high_mask_space = generate_mask(high_thresh, C, T, H, W)

    mid_high_mask_, mid_high_mask_time_, mid_high_mask_space_ = generate_mask(mid_high_thresh, C, T, H, W)
    mid_high_mask, mid_high_mask_time, mid_high_mask_space = mid_high_mask_ - high_mask, mid_high_mask_time_ - high_mask_time, mid_high_mask_space_ - high_mask_space

    mid_low_mask_, mid_low_mask_time_, mid_low_mask_space_ = generate_mask(mid_low_thresh, C, T, H, W)
    mid_low_mask, mid_low_mask_time, mid_low_mask_space = mid_low_mask_ - mid_high_mask_, mid_low_mask_time_ - mid_high_mask_time_, mid_low_mask_space_ - mid_high_mask_space_

    low_mask_, low_mask_time_, low_mask_space_ = generate_mask(low_thresh, C, T, H, W)
    low_mask, low_mask_time, low_mask_space = low_mask_ - mid_low_mask_, low_mask_time_ - mid_low_mask_time_, low_mask_space_ - mid_low_mask_space_

    # Function to extract and calculate mean power
    def extract_mean_power_mask(mag, mask):
        # Apply the mask
        masked_mag = mag * mask
        # Calculate the mean power
        mean_power = (masked_mag **2).sum() / mask.sum()
        return mean_power

    high_freq_power = extract_mean_power_mask(mag, high_mask)
    mid_high_power = extract_mean_power_mask(mag, mid_high_mask)
    mid_low_power = extract_mean_power_mask(mag, mid_low_mask)
    low_freq_power = extract_mean_power_mask(mag, low_mask)

    high_freq_power_time = extract_mean_power_mask(mag_time, high_mask_time)
    mid_high_power_time = extract_mean_power_mask(mag_time, mid_high_mask_time)
    mid_low_power_time = extract_mean_power_mask(mag_time, mid_low_mask_time)
    low_freq_power_time = extract_mean_power_mask(mag_time, low_mask_time)

    high_freq_power_space = extract_mean_power_mask(mag_space, high_mask_space)
    mid_high_power_space = extract_mean_power_mask(mag_space, mid_high_mask_space)
    mid_low_power_space = extract_mean_power_mask(mag_space, mid_low_mask_space)
    low_freq_power_space = extract_mean_power_mask(mag_space, low_mask_space)

    return high_freq_power.item(), mid_high_power.item(), mid_low_power.item(), low_freq_power.item(), high_freq_power_space.item(), mid_high_power_space.item(), mid_low_power_space.item(), low_freq_power_space.item(), high_freq_power_time.item(), mid_high_power_time.item(), mid_low_power_time.item(), low_freq_power_time.item()



def identify_high_low_frequencies(mag_space):
    # Assuming mag_space is the magnitude of the spatial FFT and has shape (C, T, H, W)
    C, T, H, W = mag_space.shape

    # Define frequency ranges
    def get_indices(start, end, length):
        return torch.arange(int(start * length), int(end * length))

    # High and low frequencies are typically at the edges of the spectrum
    high_freq_indices = (torch.cat((get_indices(0, 0.125, T), get_indices(0.875, 1, T))),
                         torch.cat((get_indices(0, 0.125, H), get_indices(0.875, 1, H))),
                         torch.cat((get_indices(0, 0.125, W), get_indices(0.875, 1, W))))

    mid_high_freq_indices = (torch.cat((get_indices(0.125, 0.25, T), get_indices(0.75, 0.875, T))),
                             torch.cat((get_indices(0.125, 0.25, H), get_indices(0.75, 0.875, H))),
                             torch.cat((get_indices(0.125, 0.25, W), get_indices(0.75, 0.875, W))))

    mid_low_freq_indices = (torch.cat((get_indices(0.25, 0.375, T), get_indices(0.625, 0.75, T))),
                            torch.cat((get_indices(0.25, 0.375, H), get_indices(0.625, 0.75, H))),
                            torch.cat((get_indices(0.25, 0.375, W), get_indices(0.625, 0.75, W))))

    low_freq_indices = (torch.cat((get_indices(0.375, 0.5, T), get_indices(0.5, 0.625, T))),
                        torch.cat((get_indices(0.375, 0.5, H), get_indices(0.5, 0.625, H))),
                        torch.cat((get_indices(0.375, 0.5, W), get_indices(0.5, 0.625, W))))

    # Function to extract and calculate mean power
    def extract_mean_power(mag, indices):
        # Extract the specific frequencies using advanced indexing
        extracted = mag[:, indices[0], :, :][:, :, indices[1], :][:, :, :, indices[2]]
        # Square the magnitudes to get power and then take mean to get average power
        mean_power = (extracted ** 2).mean()
        return mean_power

    high_freq_power = extract_mean_power(mag_space, high_freq_indices)
    mid_high_power = extract_mean_power(mag_space, mid_high_freq_indices)
    mid_low_power = extract_mean_power(mag_space, mid_low_freq_indices)
    low_freq_power = extract_mean_power(mag_space, low_freq_indices)

    return high_freq_power.item(), mid_high_power.item(), mid_low_power.item(), low_freq_power.item()


def _old_identify_high_low_frequencies(mag_space):
    # Assuming mag_space is the magnitude of the spatial FFT and has shape (C, T, H, W)
    C, T, H, W = mag_space.shape
    
    # Calculate indices for high frequencies based on thresholds
    high_freq_t_start_1 = 0
    high_freq_h_start_1 = 0
    high_freq_w_start_1 = 0
    high_freq_t_end_1 = int(0.125 * T)
    high_freq_h_end_1 = int(0.125 * H)
    high_freq_w_end_1 = int(0.125 * W)

    high_freq_t_start_2 = int(0.875 * T)
    high_freq_h_start_2 = int(0.875 * H)
    high_freq_w_start_2 = int(0.875 * W)
    high_freq_t_end_2 = T
    high_freq_h_end_2 = H
    high_freq_w_end_2 = W

    # Mid-high-frequency indices
    mid_high_freq_t_start_1 = int(0.125 * T)
    mid_high_freq_h_start_1 = int(0.125 * H)
    mid_high_freq_w_start_1 = int(0.125 * W)
    mid_high_freq_t_end_1 = int(0.25 * T)
    mid_high_freq_h_end_1 = int(0.25 * H)
    mid_high_freq_w_end_1 = int(0.25 * W)

    mid_high_freq_t_start_2 = int(0.75 * T)
    mid_high_freq_h_start_2 = int(0.75 * H)
    mid_high_freq_w_start_2 = int(0.75 * W)
    mid_high_freq_t_end_2 = int(0.875 * T)
    mid_high_freq_h_end_2 = int(0.875 * H)
    mid_high_freq_w_end_2 = int(0.875 * W)

    # Mid-low-frequency indices
    mid_low_freq_t_start_1 = int(0.25 * T)
    mid_low_freq_h_start_1 = int(0.25 * H)
    mid_low_freq_w_start_1 = int(0.25 * W)
    mid_low_freq_t_end_1 = int(0.375 * T)
    mid_low_freq_h_end_1 = int(0.375 * H)
    mid_low_freq_w_end_1 = int(0.375 * W)

    mid_low_freq_t_start_2 = int(0.625 * T)
    mid_low_freq_h_start_2 = int(0.625 * H)
    mid_low_freq_w_start_2 = int(0.625 * W)
    mid_low_freq_t_end_2 = int(0.75 * T)
    mid_low_freq_h_end_2 = int(0.75 * H)
    mid_low_freq_w_end_2 = int(0.75 * W)

    # Low-frequency indices
    low_freq_t_start_1 = int(0.375 * T)
    low_freq_h_start_1 = int(0.375 * H)
    low_freq_w_start_1 = int(0.375 * W)
    low_freq_t_end_1 = int(0.5 * T)
    low_freq_h_end_1 = int(0.5 * H)
    low_freq_w_end_1 = int(0.5 * W)

    low_freq_t_start_2 = int(0.5 * T)
    low_freq_h_start_2 = int(0.5 * H)
    low_freq_w_start_2 = int(0.5 * W)
    low_freq_t_end_2 = int(0.625 * T)
    low_freq_h_end_2 = int(0.625 * H)
    low_freq_w_end_2 = int(0.625 * W)

    # Creating index arrays for each frequency range
    high_freq_indices = (torch.cat((torch.arange(high_freq_t_start_1, high_freq_t_end_1), torch.arange(high_freq_t_start_2, high_freq_t_end_2))),
                         torch.cat((torch.arange(high_freq_h_start_1, high_freq_h_end_1), torch.arange(high_freq_h_start_2, high_freq_h_end_2))),
                         torch.cat((torch.arange(high_freq_w_start_1, high_freq_w_end_1), torch.arange(high_freq_w_start_2, high_freq_w_end_2))))
    mid_high_freq_indices = (torch.cat((torch.arange(mid_high_freq_t_start_1, mid_high_freq_t_end_1), torch.arange(mid_high_freq_t_start_2, mid_high_freq_t_end_2))),
                             torch.cat((torch.arange(mid_high_freq_h_start_1, mid_high_freq_h_end_1), torch.arange(mid_high_freq_h_start_2, mid_high_freq_h_end_2))),
                             torch.cat((torch.arange(mid_high_freq_w_start_1, mid_high_freq_w_end_1), torch.arange(mid_high_freq_w_start_2, mid_high_freq_w_end_2))))
    mid_low_freq_indices = (torch.cat((torch.arange(mid_low_freq_t_start_1, mid_low_freq_t_end_1), torch.arange(mid_low_freq_t_start_2, mid_low_freq_t_end_2))),
                            torch.cat((torch.arange(mid_low_freq_h_start_1, mid_low_freq_h_end_1), torch.arange(mid_low_freq_h_start_2, mid_low_freq_h_end_2))),
                            torch.cat((torch.arange(mid_low_freq_w_start_1, mid_low_freq_w_end_1), torch.arange(mid_low_freq_w_start_2, mid_low_freq_w_end_2))))
    low_freq_indices = (torch.cat((torch.arange(low_freq_t_start_1, low_freq_t_end_1), torch.arange(low_freq_t_start_2, low_freq_t_end_2))),
                        torch.cat((torch.arange(low_freq_h_start_1, low_freq_h_end_1), torch.arange(low_freq_h_start_2, low_freq_h_end_2))),
                        torch.cat((torch.arange(low_freq_w_start_1, low_freq_w_end_1), torch.arange(low_freq_w_start_2, low_freq_w_end_2))))

    # Extracting frequency content using multi-dimensional indexing
    high_freq_content = (mag_space[:, high_freq_indices[0], :, :][:, :, high_freq_indices[1], :][:, :, :, high_freq_indices[2]]**2).mean()
    mid_high_content = (mag_space[:, mid_high_freq_indices[0], :, :][:, :, mid_high_freq_indices[1], :][:, :, :, mid_high_freq_indices[2]]**2).mean()
    mid_low_content = (mag_space[:, mid_low_freq_indices[0], :, :][:, :, mid_low_freq_indices[1], :][:, :, :, mid_low_freq_indices[2]]**2).mean()
    low_freq_content = (mag_space[:, low_freq_indices[0], :, :][:, :, low_freq_indices[1], :][:, :, :, low_freq_indices[2]]**2).mean()

    return high_freq_content.item(), mid_high_content.item(), mid_low_content.item(), low_freq_content.item()


def batch_ddim_sampling(model, cond, noise_shape, n_samples=1, ddim_steps=50, ddim_eta=1.0,\
                        cfg_scale=1.0, temporal_cfg_scale=None, **kwargs):
    ddim_sampler = DDIMSampler(model)
    uncond_type = model.uncond_type
    batch_size = noise_shape[0]

    ## construct unconditional guidance
    if cfg_scale != 1.0:
        if uncond_type == "empty_seq":
            prompts = batch_size * [""]
            #prompts = N * T * [""]  ## if is_imgbatch=True
            uc_emb = model.get_learned_conditioning(prompts)
        elif uncond_type == "zero_embed":
            c_emb = cond["c_crossattn"][0] if isinstance(cond, dict) else cond
            uc_emb = torch.zeros_like(c_emb)
                
        ## process image embedding token
        if hasattr(model, 'embedder'):
            uc_img = torch.zeros(noise_shape[0],3,224,224).to(model.device)
            ## img: b c h w >> b l c
            uc_img = model.get_image_embeds(uc_img)
            uc_emb = torch.cat([uc_emb, uc_img], dim=1)
        
        if isinstance(cond, dict):
            uc = {key:cond[key] for key in cond.keys()}
            uc.update({'c_crossattn': [uc_emb]})
        else:
            uc = uc_emb
    else:
        uc = None
    
    x_T = None
    batch_variants = []
    #batch_variants1, batch_variants2 = [], []
    for _ in range(n_samples):
        if ddim_sampler is not None:
            kwargs.update({"clean_cond": True})
            samples, _, init_noise = ddim_sampler.sample(S=ddim_steps,
                                            conditioning=cond,
                                            batch_size=noise_shape[0],
                                            shape=noise_shape[1:],
                                            verbose=True,
                                            unconditional_guidance_scale=cfg_scale,
                                            unconditional_conditioning=uc,
                                            eta=ddim_eta,
                                            temporal_length=noise_shape[2],
                                            conditional_guidance_scale_temporal=temporal_cfg_scale,
                                            x_T=x_T,
                                            **kwargs
                                            )
        ## reconstruct from latent to pixel space
        real_freq, real_freq_space, real_freq_time = analyze_frequency_components_torch(samples.detach().cpu().float())
        high_freq_content, mid_high_freq_content, mid_low_freq_content, low_freq_content, high_freq_content_space, mid_high_freq_content_space, mid_low_freq_content_space, low_freq_content_space,  high_freq_content_time, mid_high_freq_content_time, mid_low_freq_content_time, low_freq_content_time = identify_high_low_masks(real_freq, real_freq_space, real_freq_time)

        real_freq_noise, real_freq_noise_space, real_freq_noise_time= analyze_frequency_components_torch(init_noise.detach().cpu().float())
        high_freq_content_noise, mid_high_freq_content_noise, mid_low_freq_content_noise, low_freq_content_noise, high_freq_content_space_noise, mid_high_freq_content_space_noise, mid_low_freq_content_space_noise, low_freq_content_space_noise,  high_freq_content_time_noise, mid_high_freq_content_time_noise, mid_low_freq_content_time_noise, low_freq_content_time_noise = identify_high_low_masks(real_freq_noise, real_freq_noise_space, real_freq_noise_time)

        snr_high = snr_calculation(high_freq_content, high_freq_content_noise)
        snr_mid_high = snr_calculation(mid_high_freq_content, mid_high_freq_content_noise)
        snr_mid_low = snr_calculation(mid_low_freq_content, mid_low_freq_content_noise)
        snr_low = snr_calculation(low_freq_content, low_freq_content_noise)

        snr_high_space = snr_calculation(high_freq_content_space, high_freq_content_space_noise)
        snr_mid_high_space = snr_calculation(mid_high_freq_content_space, mid_high_freq_content_space_noise)
        snr_mid_low_space = snr_calculation(mid_low_freq_content_space, mid_low_freq_content_space_noise)
        snr_low_space = snr_calculation(low_freq_content_space, low_freq_content_space_noise)

        snr_high_time = snr_calculation(high_freq_content_time, high_freq_content_time_noise)
        snr_mid_high_time = snr_calculation(mid_high_freq_content_time, mid_high_freq_content_time_noise)
        snr_mid_low_time = snr_calculation(mid_low_freq_content_time, mid_low_freq_content_time_noise)
        snr_low_time = snr_calculation(low_freq_content_time, low_freq_content_time_noise)

        

        print(f"High frequency content average: {high_freq_content}")
        print(f"Middle high frequency content average: {mid_high_freq_content}")
        print(f"Middle low frequency content average: {mid_low_freq_content}")
        print(f"Low frequency content average: {low_freq_content}")

        print(f"High frequency content average (noise): {high_freq_content_noise}")
        print(f"Middle high frequency content average (noise): {mid_high_freq_content_noise}")
        print(f"Middle low frequency content average (noise): {mid_low_freq_content_noise}")
        print(f"Low frequency content average (noise): {low_freq_content_noise}")

        print(f"SNR High: {snr_high}")
        print(f"SNR Middle High: {snr_mid_high}")
        print(f"SNR Middle Low: {snr_mid_low}")
        print(f"SNR Low: {snr_low}")

        breakpoint()


        batch_images = model.decode_first_stage_2DAE(samples) # where final results output

        batch_variants.append(batch_images)
    ## batch, <samples>, c, t, h, w
    batch_variants = torch.stack(batch_variants, dim=1)
    return batch_variants


def get_filelist(data_dir, ext='*'):
    file_list = glob.glob(os.path.join(data_dir, '*.%s'%ext))
    file_list.sort()
    return file_list

def get_dirlist(path):
    list = []
    if (os.path.exists(path)):
        files = os.listdir(path)
        for file in files:
            m = os.path.join(path,file)
            if (os.path.isdir(m)):
                list.append(m)
    list.sort()
    return list


def load_model_checkpoint(model, ckpt):
    def load_checkpoint(model, ckpt, full_strict):
        state_dict = torch.load(ckpt, map_location="cpu")
        try:
            ## deepspeed
            new_pl_sd = OrderedDict()
            for key in state_dict['module'].keys():
                new_pl_sd[key[16:]]=state_dict['module'][key]
            model.load_state_dict(new_pl_sd, strict=full_strict)
        except:
            if "state_dict" in list(state_dict.keys()):
                state_dict = state_dict["state_dict"]
            model.load_state_dict(state_dict, strict=full_strict)
        return model
    load_checkpoint(model, ckpt, full_strict=True)
    print('>>> model checkpoint loaded.')
    return model


def load_prompts(prompt_file):
    f = open(prompt_file, 'r')
    prompt_list = []
    for idx, line in enumerate(f.readlines()):
        l = line.strip()
        if len(l) != 0:
            prompt_list.append(l)
        f.close()
    return prompt_list


def load_video_batch(filepath_list, frame_stride, video_size=(256,256), video_frames=16):
    '''
    Notice about some special cases:
    1. video_frames=-1 means to take all the frames (with fs=1)
    2. when the total video frames is less than required, padding strategy will be used (repreated last frame)
    '''
    fps_list = []
    batch_tensor = []
    assert frame_stride > 0, "valid frame stride should be a positive interge!"
    for filepath in filepath_list:
        padding_num = 0
        vidreader = VideoReader(filepath, ctx=cpu(0), width=video_size[1], height=video_size[0])
        fps = vidreader.get_avg_fps()
        total_frames = len(vidreader)
        max_valid_frames = (total_frames-1) // frame_stride + 1
        if video_frames < 0:
            ## all frames are collected: fs=1 is a must
            required_frames = total_frames
            frame_stride = 1
        else:
            required_frames = video_frames
        query_frames = min(required_frames, max_valid_frames)
        frame_indices = [frame_stride*i for i in range(query_frames)]

        ## [t,h,w,c] -> [c,t,h,w]
        frames = vidreader.get_batch(frame_indices)
        frame_tensor = torch.tensor(frames.asnumpy()).permute(3, 0, 1, 2).float()
        frame_tensor = (frame_tensor / 255. - 0.5) * 2
        if max_valid_frames < required_frames:
            padding_num = required_frames - max_valid_frames
            frame_tensor = torch.cat([frame_tensor, *([frame_tensor[:,-1:,:,:]]*padding_num)], dim=1)
            print(f'{os.path.split(filepath)[1]} is not long enough: {padding_num} frames padded.')
        batch_tensor.append(frame_tensor)
        sample_fps = int(fps/frame_stride)
        fps_list.append(sample_fps)
    
    return torch.stack(batch_tensor, dim=0)

from PIL import Image
def load_image_batch(filepath_list, image_size=(256,256)):
    batch_tensor = []
    for filepath in filepath_list:
        _, filename = os.path.split(filepath)
        _, ext = os.path.splitext(filename)
        if ext == '.mp4':
            vidreader = VideoReader(filepath, ctx=cpu(0), width=image_size[1], height=image_size[0])
            frame = vidreader.get_batch([0])
            img_tensor = torch.tensor(frame.asnumpy()).squeeze(0).permute(2, 0, 1).float()
        elif ext == '.png' or ext == '.jpg':
            img = Image.open(filepath).convert("RGB")
            rgb_img = np.array(img, np.float32)
            #bgr_img = cv2.imread(filepath, cv2.IMREAD_COLOR)
            #bgr_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
            rgb_img = cv2.resize(rgb_img, (image_size[1],image_size[0]), interpolation=cv2.INTER_LINEAR)
            img_tensor = torch.from_numpy(rgb_img).permute(2, 0, 1).float()
        else:
            print(f'ERROR: <{ext}> image loading only support format: [mp4], [png], [jpg]')
            raise NotImplementedError
        img_tensor = (img_tensor / 255. - 0.5) * 2
        batch_tensor.append(img_tensor)
    return torch.stack(batch_tensor, dim=0)


def save_videos(batch_tensors, savedir, filenames, fps=10):
    # b,samples,c,t,h,w
    n_samples = batch_tensors.shape[1]
    for idx, vid_tensor in enumerate(batch_tensors):
        video = vid_tensor.detach().cpu()
        video = torch.clamp(video.float(), -1., 1.)
        video = video.permute(2, 0, 1, 3, 4) # t,n,c,h,w
        frame_grids = [torchvision.utils.make_grid(framesheet, nrow=int(n_samples)) for framesheet in video] #[3, 1*h, n*w]
        grid = torch.stack(frame_grids, dim=0) # stack in temporal dim [t, 3, n*h, w]
        grid = (grid + 1.0) / 2.0
        grid = (grid * 255).to(torch.uint8).permute(0, 2, 3, 1)
        savepath = os.path.join(savedir, f"{filenames[idx]}_combine_0.25.mp4")
        torchvision.io.write_video(savepath, grid, fps=fps, video_codec='h264', options={'crf': '10'})