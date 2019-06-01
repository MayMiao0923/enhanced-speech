import numpy as np


def enframe(x, window_size, hop_size):
    """Enframe 1d-array to 2d-array.  将1d阵列框架化为2d阵列。
    
    Args:
      x: 1darray
      window_size: int
      hop_size: int
    
    Returns:
      frames: 2d-array, (num_frames, window_size)
    """
    num_bytes = x.strides[-1]   # Number of bytes 字节数
    seq_len = len(x)
    num_frames = calculate_num_of_frames(seq_len, window_size, hop_size)
    
    # Pad
    new_len = window_size + (num_frames - 1) * hop_size
    pad_len = new_len - seq_len
    x = np.pad(array=x, 
               pad_width=(0, pad_len), 
               mode='constant', 
               constant_values=0.)
    
    # Enframe 分帧
    frames = np.lib.stride_tricks.as_strided(
        x=x, 
        shape=(num_frames, window_size), 
        strides=(hop_size * num_bytes, num_bytes))
    
    frames = np.copy(frames)
    return frames
    
    
def calculate_num_of_frames(seq_len, window_size, hop_size):
    """Calculate number of frames of a 1d-array.  计算1d阵列的帧数。
    
    Args:
      seq_len: int, length of 1d-array. 
      window_size: int
      hop_size: int, number of frames to hop. 
    """
    num_frames = int(np.ceil((seq_len - window_size) / float(hop_size))) + 1
    return num_frames


def stft(x, window_size, hop_size, window, mode='complex'):
    """Short time Fourier transform. STFT
    
    Args:
      x: 1darray
      window_size: int
      hop_size: int
      window: 1darray, e.g. np.hamming(window_size)
      mode: str, 'complex' | 'magnitude'
      
    Returns:
      sp: 2darray, spectrogram
    """
    assert window_size == len(window), "len(window) must equal to window_size!"
    
    # Enframe 分帧
    frames = enframe(x, window_size, hop_size)
    
    # Apply window 应用窗口
    frames *= window[None, :]

    # STFT
    sp = np.fft.rfft(frames, axis=-1, norm='ortho')
    
    if mode == 'complex':
        return sp
        
    elif mode == 'magnitude':
        return np.abs(sp)
    
    else:
        raise Exception("Incorrect mode!")


def istft(x):
    """Inverse short time Fourier transform of a 2d-array. ISTFT
    
    Args:
      x: 2d-array, Complex spectrum. 
      
    Returns:
      frames: 2d-array. 
    """
    
    frames = np.fft.irfft(x, axis=-1, norm='ortho')
    return frames


def real_to_complex(mag_sp, cmplx_sp):
    """Convert magnitude spectrogrom to complex spectrum. 将幅度谱转换为复杂谱
    
    Args:
      mag_sp: 2d-array, magnitidue spectrogram. 
      cmplx_sp: 2d-array, reference complex spectrum to extract angle. 
      
    Returns:
      2d-array, recovered complex spectrum. 
    """
    theta = np.angle(cmplx_sp)
    recovered_sp = mag_sp * np.exp(1j * theta)
    return recovered_sp


def get_cola_constant(hop_size, window):
    """Check if constant overlap add is satisifed. Return the COLA constant. 检查是否满足常量重叠添加。 返回COLA常数。
    
    Args:
      hop_size: int
      window: 1darray, window array, e.g., np.hamming(window_size). 
      
    Returns:
      cola_constant: float. Used later for overlap add. 
    """
    
    window_size = len(window)
    assert window_size % hop_size == 0, "COLA is not satisfied!"
    
    n = window_size / hop_size
    buffer = np.zeros(window_size)

    # Sum
    for i1 in xrange(n):
        buffer += np.roll(window, i1 * hop_size)
        
    # Check if sum of window is constant 检查窗口总和是否恒定
    assert np.max(buffer) - np.min(buffer) < 0.01, "COLA is not satisfied"
        
    cola_constant = np.mean(buffer)    
    return cola_constant
    

def overlap_add(frames, hop_size, cola_constant):
    """Overlap add of recovered frames. 重叠添加恢复的帧
    
    Args:
      frames: 2darray, recovered frames, (num_frames, window_size). 
      hop_size, int
      cola_constant: float
      
    Returns:
      seq: 1darray, overlap added sequence. 
    """
    
    (num_frames, window_size) = frames.shape
    
    # Overlap add
    seq_len = window_size + (num_frames - 1) * hop_size
    seq = np.zeros(seq_len)
    
    for n in xrange(num_frames):
        seq[n * hop_size : n * hop_size + window_size] += frames[n]
        
    # Normalize 规范化
    seq /= cola_constant
        
    return seq
    

if __name__ == '__main__':
    sample_rate = 32000
    window_size = 1024
    hop_size = 512
    window = np.hamming(window_size)
    
    t = np.arange(10000)
    audio = np.cos(440 * 2 * np.pi * t / float(sample_rate))
    
    # STFT
    sp = stft(audio, window_size, hop_size, window)
    mag_sp = np.abs(sp)

    # Recover complex spectrum from magnitude spectrogram 从幅度谱图中恢复复杂的光谱
    recovered_sp = real_to_complex(mag_sp, sp)
    
    # ISTFT
    frames = istft(recovered_sp)
    
    # Overlap add 重叠添加
    cola_constant = get_cola_constant(hop_size, window)
    seq = overlap_add(frames, hop_size, cola_constant)
    
    import matplotlib.pyplot as plt
    
    fig, axs = plt.subplots(2,1)
    axs[0].plot(audio)
    axs[1].plot(seq)
    plt.show()