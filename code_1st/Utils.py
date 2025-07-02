import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from matplotlib.ticker import MultipleLocator
from scipy.signal import butter, filtfilt, resample_poly, medfilt, hilbert, get_window, find_peaks, welch
from scipy.ndimage import uniform_filter1d
from scipy.interpolate import interp1d, UnivariateSpline
import pywt

# リサンプリング
def resample_waveform(data: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """
    単純なリサンプリング(ポリフェーズフィルタ)を行う。
    scipy.signal.resample_poly などを利用し、
    target_sr / orig_sr でリサンプリングする。
    """
    # 例: orig_sr=8000, target_sr=4000 ⇒ down-sample 1/2
    # 例: orig_sr=44100, target_sr=4000 ⇒ 単純計算で約1/11
    # もっと高度なリサンプリングには librosa.resample なども利用可
    # ここでは整数レートを想定。小数の場合は比率計算が必要
    if orig_sr == target_sr:
        return data
    # たとえば factor = target_sr / orig_sr
    # resample_poly は up, down = target_sr, orig_sr という使い方をする
    up = target_sr
    down = orig_sr
    resampled = resample_poly(data, up, down)
    return resampled

##### フィルタ #####
# スパイク除去
def spike_removal(data, window_size=5):
    # 中央値フィルタでスパイク除去
    y = medfilt(data, window_size)
    return y

# ハイパスフィルタ
def hpf(data, sr, fc_high, order=4):
    """
    data: フィルタをかける1次元配列
    sr: サンプリングレート
    fc_high: 上限周波数（Hz）
    order: フィルタの次数
    """
    nyq = 0.5 * sr  # ナイキスト周波数
    high = fc_high / nyq
    b, a = butter(order, high, btype='high')
    
    y = filtfilt(b, a, data)
    return y

# ローパスフィルタ
def lpf(data, sr, fc_low, order=4):
    """
    data: フィルタをかける1次元配列
    sr: サンプリングレート
    fc_low: 下限周波数（Hz）
    order: フィルタの次数
    """
    nyq = 0.5 * sr  # ナイキスト周波数
    low = fc_low / nyq
    b, a = butter(order, low, btype='low')
    
    y = filtfilt(b, a, data)
    return y

# バンドパスフィルタ
def bandpass(data, sr, fc_low, fc_high, order=4):
    """
    data: フィルタをかける1次元配列
    sr: サンプリングレート
    fc_low: 下限周波数（Hz）
    fc_high: 上限周波数（Hz）
    order: フィルタの次数
    """
    nyq = 0.5 * sr  # ナイキスト周波数
    low = fc_low / nyq
    high = fc_high / nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

# 移動平均
def moving_average(data, window_size):
    window_size = int(round(window_size))  # 小数点以下を四捨五入し整数型にする
    if window_size < 3:
        return data
    elif window_size % 2 == 0:
        window_size += 1  # 偶数の場合は奇数にする(繰り上げ)
    return uniform_filter1d(data, window_size, mode='reflect')

# 音量エンベロープ
def get_sound_vol(data, fs, window_size=11):
    analytic_signal = hilbert(data)  # ヒルベルト変換
    envelope = np.abs(analytic_signal)  # 振幅エンベロープ
    envelope_smooth = moving_average(envelope, window_size)  # 移動平均
    vol_prom_coeff = 3  # ピーク検出のprominence係数
    vol_distance = int(0.04 * fs)  # ピーク間の最小距離(0.05秒)
    vol_peaks, _ = find_peaks(envelope_smooth, 
                                prominence=vol_prom_coeff * np.std(envelope_smooth),
                                distance=vol_distance)
    vol_mean = round(np.mean(envelope_smooth[vol_peaks]),0)  # ピークの平均値を音量とする
    return vol_mean

# ガロッテしきい値
def garrote_thresholding(coef, threshold):
    # 閾値をスケールごとに適応
    return coef * (1 - (threshold**2 / (coef**2 + 1e-10))) * (np.abs(coef) > threshold)

# ウェーブレットノイズ除去
def wavelet_denoising(data, wavelet='bior6.8', k_thr=0.45, max_level=7):
    """
    ウェーブレット分解を用いたノイズ除去
    Parameters:
    -----------
    data : np.ndarray
        入力信号
    wavelet : str
        ウェーブレットの種類（デフォルト: 'bior6.8'）
    k_thr : float
        しきい値の係数（デフォルト: 0.45）
    max_level : int
        最大分解レベル（デフォルト: 7）
    
    Returns:
    --------
    denoised_signal : np.ndarray
        ノイズ除去された信号
    coeffs_info : dict
        分解情報（デバッグ用）
    """
    # ---------- STEP 1: ウェーブレット分解 ----------
    level = min(pywt.dwt_max_level(len(data), pywt.Wavelet(wavelet).dec_len), max_level)
    coeffs = pywt.wavedec(data, wavelet, level=level)

    # ---------- STEP 2: しきい値処理（ガロッテしきい値） ----------
    # しきい値適用（スケールごと）
    coeffs_thresh = [coeffs[0]]  # 近似係数はそのまま保持
    for j in range(1, len(coeffs)):
        sigma_i = np.median(np.abs(coeffs[j])) / 0.6745
        thr_i = k_thr * sigma_i * np.sqrt(2 * np.log(len(coeffs[j])))  # 控えめ + ガロッテ
        coeffs_thresh.append(garrote_thresholding(coeffs[j], thr_i))

    # ---------- STEP 3: 逆変換（再構成） ----------
    denoised_signal = pywt.waverec(coeffs_thresh, wavelet)
    
    # デバッグ情報
    coeffs_info = {
        'wavelet': wavelet,
        'level': level,
        'k_thr': k_thr,
        'original_coeffs': coeffs,
        'thresholded_coeffs': coeffs_thresh
    }
    
    return denoised_signal, coeffs_info

# フェードイン/アウト
def apply_fade_window(signal, fs, fade_sec):
    fade_len = int(fade_sec * fs)
    window = np.ones(len(signal))
    fade_in = np.linspace(0, 1, fade_len)
    fade_out = np.linspace(1, 0, fade_len)
    window[:fade_len] = fade_in
    window[-fade_len:] = fade_out
    return signal * window

# === RMSエンベロープの計算 ===
def rms_envelope(signal, fs, frame_size_sec, hop_size_sec):
    frame_size = int(frame_size_sec * fs)
    hop_size = int(hop_size_sec * fs)
    rms = []
    for i in range(0, len(signal) - frame_size + 1, hop_size):
        frame = signal[i:i+frame_size]
        rms.append(np.sqrt(np.mean(frame ** 2)))
    return np.array(rms)

# === 補間で元のサンプル数に戻す ===
def interpolate_rms(signal, fs, hop_size, method="spline"):
    signal_len = len(signal)  # サンプル数
    duration = signal_len * hop_size  # 秒
    # 元の時間軸
    t_orig = np.arange(signal_len) * hop_size
    t_new = np.arange(0, duration, 1/fs)

    if method == "linear":
        interp_func = interp1d(t_orig, signal, kind='linear', fill_value="extrapolate")
    elif method == "spline":
        interp_func = UnivariateSpline(t_orig, signal, s=5, ext='const')  # s=0:全点通過
    else:
        raise ValueError("method must be 'linear' or 'spline'")

    # 補間実行
    interpolated_rms = interp_func(t_new)
    return interpolated_rms
    
# === AGC-stage1 ===
def agc(rms_env, agc_params):
    gain = 1.0
    gains = []
    target_level = agc_params['target_level']
    attack = agc_params['attack']
    release = agc_params['release']
    min_gain = agc_params['min_gain']
    max_gain = agc_params['max_gain']

    for level in rms_env:
        desired = target_level / (level + 1e-8)
        if desired < gain:
            gain = attack * gain + (1 - attack) * desired
        else:
            gain = release * gain + (1 - release) * desired
        gain = np.clip(gain, min_gain, max_gain)
        gains.append(gain)
    return np.array(gains)

# === Apply Gain to Signal ===
def apply_gain(signal, gains, fs, hop_size_sec):
    hop_size = int(hop_size_sec * fs)
    x_env = np.arange(0, len(gains)) * hop_size
    x_sig = np.arange(len(signal))
    interp_func = interp1d(x_env, gains, kind='linear', bounds_error=False, fill_value='extrapolate')
    gain_applied = interp_func(x_sig)
    return signal * gain_applied

# RMS-AGC 2-stages
def rms_agc_2stages(signal, fs, fade_sec=0.5, hop_size_sec=0.01, frame_size_sec=0.05, 
                   ma_window_size=55, agc1_params=None, agc2_params=None):
    """
    RMS-AGC 2-stages処理を一括で実行
    
    Parameters:
    -----------
    signal : np.ndarray
        入力信号（ウェーブレットデノイズ後など）
    fs : int
        サンプリングレート
    fade_sec : float
        フェードイン/アウト時間（秒）
    hop_size_sec : float
        ホップサイズ（秒）
    frame_size_sec : float
        フレームサイズ（秒）
    ma_window_size : int
        移動平均のウィンドウサイズ
    agc1_params : dict
        AGC-stage1のパラメータ
    agc2_params : dict
        AGC-stage2のパラメータ
    
    Returns:
    --------
    rms_env2_interp : np.ndarray
        補間されたRMSエンベロープ(FFT処理用)
    rms_env2_ma : np.ndarray
        最終的な処理済みRMSエンベロープ
    """
    
    # デフォルトパラメータの設定
    if agc1_params is None:
        agc1_params = {'target_level': 0.1, 
                       'attack': 0.8, 
                       'release': 0.9, 
                       'min_gain': 0.05, 
                       'max_gain': 10.0}
    if agc2_params is None:
        agc2_params = {'target_level': 0.1, 
                       'attack': 0.5, 
                       'release': 0.8, 
                       'min_gain': 0.1, 
                       'max_gain': 5.0}
    
    # フェードイン/アウト
    denoised_signal = apply_fade_window(signal, fs, fade_sec)
    
    # STEP 1: RMS envelope
    rms_env0 = rms_envelope(denoised_signal, fs, frame_size_sec, hop_size_sec)
    rms_env0_interp = interpolate_rms(rms_env0, fs, hop_size_sec, method="linear")

    # STEP 2: AGC-stage1 (緩やかゲイン)
    gains1 = agc(rms_env0_interp, agc_params=agc1_params)
    pcg_agc1 = apply_gain(denoised_signal, gains1, fs, hop_size_sec)

    # STEP 3: AGC-stage2 (速応型ゲイン)
    gains2 = agc(rms_env0_interp, agc_params=agc2_params)
    pcg_agc2 = apply_gain(pcg_agc1, gains2, fs, hop_size_sec)

    # STEP 4: 最終的なRMSエンベロープ
    rms_env2 = rms_envelope(pcg_agc2, fs, frame_size_sec, hop_size_sec)
    rms_env2_interp = interpolate_rms(rms_env2, fs, hop_size_sec, method="linear")
    
    # RMSエンベロープに移動平均処理
    rms_env2_ma = moving_average(rms_env2_interp, ma_window_size)
    
    return rms_env2_interp, rms_env2_ma


# コンプレッション
def compress_peak_power(rms_env, threshold=0.7, ratio=0.5):
    """
    threshold: この値より大きい部分にだけコンプレッサーをかける
    ratio: 圧縮比（0.5なら2:1圧縮）
    """
    compressed = np.copy(rms_env)
    over_threshold = compressed > threshold
    compressed[over_threshold] = threshold + (compressed[over_threshold] - threshold) * ratio
    return compressed / np.max(compressed + 1e-8)  # 再正規化



# スライディングウィンドウピーク検出
def sliding_window_peak_detection(signal, fs, window_size_sec=2, step_size_sec=1,
                                min_lag_bpm_normal=120, min_lag_bpm_tachy=220,
                                prom_coeff_normal=2.5, prom_coeff_tachy=0.05,
                                height_qxx=50, height_coeff_normal=1.5, height_coeff_tachy=0.5,
                                plot=False):
    """
    スライディングウィンドウを使用してピーク検出を実行する関数
    
    Parameters:
    -----------
    signal : np.ndarray
        入力信号（コンプレッション処理後のRMSエンベロープなど）
    fs : int
        サンプリングレート
    window_size_sec : float
        窓サイズ（秒）（デフォルト: 2）
    step_size_sec : float
        ステップサイズ（秒）（デフォルト: 1）
    min_lag_bpm_normal : float
        normal用の最小lag（bpm）（デフォルト: 120）
    min_lag_bpm_tachy : float
        tachy用の最小lag（bpm）（デフォルト: 220）
    prom_coeff_normal : float
        normal用のprominence係数（デフォルト: 2.5）
    prom_coeff_tachy : float
        tachy用のprominence係数（デフォルト: 0.05）
    height_qxx : float
        heightのパーセンタイル値（デフォルト: 50）
    height_coeff_normal : float
        normal用のheight係数（デフォルト: 1.5）
    height_coeff_tachy : float
        tachy用のheight係数（デフォルト: 0.5）
    plot_enable : bool
        プロット表示フラグ（デフォルト: False）
    
    Returns:
    --------
    results : dict
        ピーク検出結果の辞書
        - avg_peak_count_normal : list
        - time_peaks_normal : list
        - avg_peak_count_tachy : list
        - time_peaks_tachy : list
    """
    
    # 時間係数の計算
    min_lag_coeff_normal = 1/(min_lag_bpm_normal/60)
    min_lag_coeff_tachy = 1/(min_lag_bpm_tachy/60)
    
    # window & data setting
    x = np.array(signal)
    window_size = int(fs * window_size_sec)
    step_size = int(fs * step_size_sec)
    num_windows = (len(x) - window_size) // step_size + 1

    avg_peak_count_normal = []  # 窓ごとの平均ピーク間隔格納用
    time_peaks_normal = []  # 窓ごとのピーク時間格納用
    avg_peak_count_tachy = []  # 窓ごとの平均ピーク間隔格納用
    time_peaks_tachy = []  # 窓ごとのピーク時間格納用

    # 窓ごとにピーク検出を実行
    if plot:
        fig, ax = plt.subplots(num_windows, figsize=(12, 2*num_windows))
        if num_windows == 1:
            ax = [ax]  # 単一窓の場合はリストに変換
            
    for l in range(num_windows):
        start = l * step_size
        end = start + window_size
        window_data = x[start:end]  # 再正規化なしで切り出し
        window_time = np.arange(start, end) / fs

        # ---ピーク検出のパラメータ---
        # 中央値絶対偏差を計算
        val = np.median(np.abs(window_data - np.median(window_data)))
        auto_prom_normal = prom_coeff_normal * val
        auto_prom_tachy = prom_coeff_tachy * val

        val_std = np.std(window_data)  # 標準偏差
        val_percentile = np.percentile(window_data, height_qxx)  # パーセンタイル
        auto_height_normal = val_percentile + (height_coeff_normal * val_std)
        auto_height_tachy = val_percentile + (height_coeff_tachy * val_std)

        # ピーク検出
        peaks_normal, _ = find_peaks(window_data,
                                   distance=fs * min_lag_coeff_normal,
                                   prominence=auto_prom_normal,
                                   height=auto_height_normal
                                   )
        peaks_tachy, _ = find_peaks(window_data,
                                  distance=fs * min_lag_coeff_tachy,
                                  prominence=auto_prom_tachy,
                                  height=auto_height_tachy
                                  )
                
        # プロットが有効な場合のみ描画
        if plot:
            ax[l].plot(window_time, window_data, color='blue', label='data')
            ax[l].scatter(window_time[peaks_normal], window_data[peaks_normal], 
                         color='green', marker='o', label='peaks_normal')
            ax[l].scatter(window_time[peaks_tachy], window_data[peaks_tachy], 
                         color='red', marker='x', label='peaks_tachy')
            ax[l].set_ylabel('Amplitude')
            ax[l].grid(True)
            ax[l].legend(loc='upper right')
            ax[l].set_title(f'Window {l+1}')
    
        # ピーク間隔の計算(normal)
        if len(peaks_normal) >= 2:
            # 2個以上：間隔計算 + 時刻記録
            intervals_normal = np.diff(peaks_normal)  # ピーク間隔の計算（単位：サンプル）
            intervals_sec_normal = intervals_normal / fs  # サンプル数を秒に変換
            avg_interval_normal = np.mean(intervals_sec_normal)  # 平均ピーク間隔
            avg_peak_count_normal.append(avg_interval_normal)  # 平均ピーク間隔をリストに追加
            peak_times_normal = peaks_normal / fs + start/fs  # ピークの時刻を秒単位で計算
            time_peaks_normal.append(peak_times_normal)  # ピークの時刻をリストに追加
        elif len(peaks_normal) == 1:
            # 1個のみ：時刻記録のみ
            peak_times_normal = peaks_normal / fs + start/fs  # ピークの時刻を秒単位で計算
            time_peaks_normal.append(peak_times_normal)  # ピークの時刻をリストに追加
            avg_peak_count_normal.append(np.nan)  # ピークがない場合はNaNを追加
        else:
            # 0個：何もなし
            avg_peak_count_normal.append(np.nan)  # ピークがない場合はNaNを追加
            time_peaks_normal.append(np.nan)  # ピークの時刻をリストに追加

        # ピーク間隔の計算(tachy)
        if len(peaks_tachy) >= 2:
            intervals_tachy = np.diff(peaks_tachy)  # ピーク間隔の計算（単位：サンプル）
            intervals_sec_tachy = intervals_tachy / fs  # サンプル数を秒に変換
            avg_interval_tachy = np.mean(intervals_sec_tachy)  # 平均ピーク間隔
            avg_peak_count_tachy.append(avg_interval_tachy)  # 平均ピーク間隔をリストに追加
            peak_times_tachy = peaks_tachy / fs + start/fs  # ピークの時刻を秒単位で計算
            time_peaks_tachy.append(peak_times_tachy)  # ピークの時刻をリストに追加
        elif len(peaks_tachy) == 1:
            # 1個のみ：時刻記録のみ
            peak_times_tachy = peaks_tachy / fs + start/fs  # ピークの時刻を秒単位で計算
            time_peaks_tachy.append(peak_times_tachy)  # ピークの時刻をリストに追加
            avg_peak_count_tachy.append(np.nan)  # ピークがない場合はNaNを追加
        else:
            # 0個：何もなし
            avg_peak_count_tachy.append(np.nan)  # ピークがない場合はNaNを追加
            time_peaks_tachy.append(np.nan)  # ピークの時刻をリストに追加

    if plot:
        plt.tight_layout()
        plt.show()

    # 結果を辞書として返す
    results = {
        'avg_peak_count_normal': avg_peak_count_normal,
        'time_peaks_normal': time_peaks_normal,
        'avg_peak_count_tachy': avg_peak_count_tachy,
        'time_peaks_tachy': time_peaks_tachy
    }
    
    return results

def peak_interval_estimation(avg_peak_count_normal, avg_peak_count_tachy):
    """
    ピーク間隔の推定を実行する関数
    
    Parameters:
    -----------
    avg_peak_count_normal : list
        normal用の平均ピーク間隔
    avg_peak_count_tachy : list
        tachy用の平均ピーク間隔
    """
    # normal
    if avg_peak_count_normal and len(avg_peak_count_normal) > 0:  # 配列が空でないことを確認
        avg_peak_count_normal = np.array(avg_peak_count_normal)
        avg_peak_count_normal = avg_peak_count_normal[~np.isnan(avg_peak_count_normal)]  
        
        if len(avg_peak_count_normal) > 0:  # NaN除去後も配列が空でないことを確認
            sorted_peaks_normal = np.sort(avg_peak_count_normal)
            hr_candidates_normal = np.round(60 / sorted_peaks_normal).astype(int)
            # 多数決を取る
            unique_hr_normal, counts_normal = np.unique(hr_candidates_normal, return_counts=True)
            # 最頻値が複数ある場合は中央値を取る
            max_count_normal = np.max(counts_normal)
            if np.sum(counts_normal == max_count_normal) > 1:
                hr_interval_normal = int(np.median(hr_candidates_normal))
            else:
                hr_interval_normal = unique_hr_normal[np.argmax(counts_normal)]
        else:
            hr_interval_normal = np.nan  # ピークがない場合はNaNを返す
    else:
        hr_interval_normal = np.nan  # ピークがないかつ配列が空の場合はNaNを返す

    # tachy
    if avg_peak_count_tachy and len(avg_peak_count_tachy) > 0:  # 配列が空でないことを確認
        avg_peak_count_tachy = np.array(avg_peak_count_tachy)
        avg_peak_count_tachy = avg_peak_count_tachy[~np.isnan(avg_peak_count_tachy)]  
        
        if len(avg_peak_count_tachy) > 0:  # NaN除去後も配列が空でないことを確認
            sorted_peaks_tachy = np.sort(avg_peak_count_tachy)
            hr_candidates_tachy = np.round(60 / sorted_peaks_tachy).astype(int)
            # 多数決を取る
            unique_hr_tachy, counts_tachy = np.unique(hr_candidates_tachy, return_counts=True)
            # 最頻値が複数ある場合は中央値を取る
            max_count_tachy = np.max(counts_tachy)
            if np.sum(counts_tachy == max_count_tachy) > 1:
                hr_interval_tachy = int(np.median(hr_candidates_tachy))
            else:
                hr_interval_tachy = unique_hr_tachy[np.argmax(counts_tachy)]
        else:
            hr_interval_tachy = np.nan  # ピークがない場合はNaNを返す
    else:
        hr_interval_tachy = np.nan  # ピークがないかつ配列が空の場合はNaNを返す

    return hr_interval_normal, hr_interval_tachy

def peak_count_estimation(time_peaks_normal, time_peaks_tachy, signal, fs):
    """
    ピーク数の推定を実行する関数
    
    Parameters:
    -----------
    time_peaks_normal : list
        normal用のピーク時間
    time_peaks_tachy : list
        tachy用のピーク時間
    signal : np.ndarray
        RMSエンベロープ
    fs : int
        サンプリングレート
    """
    # データの長さを計算し、秒単位に変換
    duration = len(signal) / fs
    
    # normal
    if time_peaks_normal and len(time_peaks_normal) > 0:
        # time_peaksを1次元配列に変換
        time_peaks_flat_normal = []
        for peaks_normal in time_peaks_normal:
            if isinstance(peaks_normal, np.ndarray):
                time_peaks_flat_normal.extend(np.round(peaks_normal, decimals=3))
            elif not np.isnan(peaks_normal):
                time_peaks_flat_normal.append(np.round(peaks_normal, decimals=3))
        time_peaks_flat_normal = np.array(time_peaks_flat_normal)
        unique_time_peaks_normal = np.unique(time_peaks_flat_normal)  # ユニーク値の抽出
        hr_count_normal = int(round(len(unique_time_peaks_normal) * 60 / duration))
    else:
        hr_count_normal = np.nan
    
    # tachy
    if time_peaks_tachy and len(time_peaks_tachy) > 0:
        time_peaks_flat_tachy = []
        for peaks_tachy in time_peaks_tachy:
            if isinstance(peaks_tachy, np.ndarray):
                time_peaks_flat_tachy.extend(np.round(peaks_tachy, decimals=3))
            elif not np.isnan(peaks_tachy):
                time_peaks_flat_tachy.append(np.round(peaks_tachy, decimals=3))
        time_peaks_flat_tachy = np.array(time_peaks_flat_tachy)
        unique_time_peaks_tachy = np.unique(time_peaks_flat_tachy)  # ユニーク値の抽出
        hr_count_tachy = int(round(len(unique_time_peaks_tachy) * 60 / duration))
    else:
        hr_count_tachy = np.nan

    return hr_count_normal, hr_count_tachy

def peak_variation_estimation(time_peaks_normal, time_peaks_tachy):
    """
    ピーク間隔のばらつきを計算する関数
    Parameters:
    time_peaks_normal : list or np.ndarray
        正常時のピーク時間配列
    time_peaks_tachy : list or np.ndarray
        頻脈時のピーク時間配列
    
    Returns:
    dict
        正常時と頻脈時の統計量を含む辞書
    """
    
    def calculate_statistics(time_peaks):
        """
        ピーク間隔の統計量を計算する内部関数
        """
        if not time_peaks or len(time_peaks) < 2:
            return {
                'std': np.nan,
                'variance': np.nan,
                'mad': np.nan
            }
        try:
            # 入力データを1次元配列に変換
            if isinstance(time_peaks, list):
                # ネストしたリストを平坦化
                flat_peaks = []
                for item in time_peaks:
                    if isinstance(item, (list, np.ndarray)):
                        flat_peaks.extend(item)
                    else:
                        flat_peaks.append(item)
                time_peaks = np.array(flat_peaks)
            else:
                time_peaks = np.array(time_peaks).flatten()
            
            # オブジェクト配列の場合の特別処理
            if time_peaks.dtype == np.dtype('O'):
                # オブジェクト配列を数値配列に変換
                numeric_peaks = []
                for item in time_peaks:
                    try:
                        if isinstance(item, (int, float, np.number)):
                            numeric_peaks.append(float(item))
                        elif isinstance(item, str):
                            numeric_peaks.append(float(item))
                        elif hasattr(item, '__iter__'):
                            # イテラブルなオブジェクトの場合
                            for subitem in item:
                                if isinstance(subitem, (int, float, np.number)):
                                    numeric_peaks.append(float(subitem))
                    except (ValueError, TypeError):
                        continue
                time_peaks = np.array(numeric_peaks)
            
            # 数値以外の要素を除去
            time_peaks = time_peaks[np.isfinite(time_peaks)]
            # 重複を削除（ユニーク値のみ抽出）
            time_peaks = np.unique(time_peaks)
            
            if len(time_peaks) < 2:
                return {
                    'std': np.nan,
                    'var': np.nan,
                    'mad': np.nan
                }
            # ピーク間隔を計算
            intervals = np.diff(time_peaks)
            # 統計量を計算
            std = np.std(intervals)  # 標準偏差
            var = np.var(intervals)  # 分散
            mad = np.mean(np.abs(intervals - np.mean(intervals)))  # 平均絶対偏差
            
            return {
                'std': std,
                'var': var,
                'mad': mad
            }
            
        except Exception as e:
            print(f"Error in calculate_statistics: {e}")
            return {
                'std': np.nan,
                'variance': np.nan,
                'mad': np.nan
            }
    
    # 正常時の統計量を計算
    normal_stats = calculate_statistics(time_peaks_normal)
    # 頻脈時の統計量を計算
    tachy_stats = calculate_statistics(time_peaks_tachy)
    
    # 結果を辞書として返す
    results = {
        'normal': normal_stats,
        'tachy': tachy_stats
    }
    return results

# FFTピーク検出と特徴量抽出
def detect_peaks_with_FFT(signal, fs, desired_sec=15, high_freq_thr=5.5, power_thr=0.3, 
                         prominence_coeff=0.6, height_threshold=0.01, plot=False, 
                         file_name="", vol_mean=0, actual_hr=0):
    """
    FFTピーク検出と特徴量抽出を実行する関数
    
    Parameters:
    -----------
    signal : np.ndarray
        入力信号（RMSエンベロープなど）
    fs : int
        サンプリングレート
    desired_sec : float
        PSD推定用のデータ秒数（デフォルト: 15）
    high_freq_thr : float
        高周波とみなす閾値（Hz）（デフォルト: 5.5）
    power_thr : float
        高周波パワーの閾値（デフォルト: 0.3）
    prominence_coeff : float
        ピーク検出のprominence係数（デフォルト: 0.6）
    height_threshold : float
        ピーク検出の最小高さ（デフォルト: 0.01）
    plot : bool
        プロット表示フラグ（デフォルト: False）
    file_name : str
        ファイル名（プロット表示時に使用）
    vol_mean : float
        音量平均値（プロット表示時に使用）
    actual_hr : int
        実際の心拍数（プロット表示時に使用）
    
    Returns:
    --------
    features : dict
        抽出された特徴量の辞書
    """
    
    desired_nperseg = int(fs * desired_sec)
    signal_length = len(signal)
    
    # ゼロパディング
    if signal_length < desired_nperseg:
        pad_total = desired_nperseg - signal_length
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        padded_signal = np.pad(signal, (pad_left, pad_right), mode='constant')
    else:
        padded_signal = signal

    # 窓関数の適用
    fft_window = get_window('hamming', len(padded_signal))
    fft_signal = padded_signal * fft_window
    
    # Welch法によるPSD推定
    f, Pxx = welch(fft_signal, fs=fs, nperseg=desired_nperseg)
    mask = (f >= 0.66) & (f <= 8.0)  # 周波数範囲を制限。0.66Hz=40bpm, 8.0Hz=480bpm
    f_sel = f[mask]  # マスクされた周波数
    P_sel = Pxx[mask]  # マスクされたパワー
    P_sel = P_sel / (np.max(P_sel) + 1e-8)  # パワーを正規化
    std_P_sel = np.std(P_sel)

    # 周波数ピーク検出
    peaks, _ = find_peaks(P_sel, prominence=std_P_sel * prominence_coeff, height=height_threshold)
    
    # ピーク情報を辞書として格納
    FFT_peaks_powers = []
    if len(peaks) > 0:
        for j in range(len(peaks)):
            FFT_peaks_powers.append({
                'freq': f_sel[peaks[j]],  # ピークの周波数
                'bpm': f_sel[peaks[j]] * 60,  # ピークのbpm
                'power': P_sel[peaks[j]]  # ピークのパワー
            })
    else:
        FFT_peaks_powers = [{'freq': 0, 'bpm': 0, 'power': 0}]  # ピークがない場合は0を格納
    
    # 特徴量抽出
    if len(FFT_peaks_powers) > 0 and FFT_peaks_powers[0]['bpm'] > 0:
        bpm_first_peak = FFT_peaks_powers[0]['bpm']
        bpm_dominant_peak = max(FFT_peaks_powers, key=lambda x: x['power'])['bpm']
        power_first_peak = FFT_peaks_powers[0]['power']
        power_dominant_peak = max(FFT_peaks_powers, key=lambda x: x['power'])['power']
        max_power_under_110 = max(FFT_peaks_powers, key=lambda x: x['power'] if x['bpm'] <= 110 else 0)['power']
        max_power_under_220 = max(FFT_peaks_powers, key=lambda x: x['power'] if 110 < x['bpm'] <= 220 else 0)['power']
        max_power_under_330 = max(FFT_peaks_powers, key=lambda x: x['power'] if 220 < x['bpm'] <= 330 else 0)['power']
        max_power_over_330 = max(FFT_peaks_powers, key=lambda x: x['power'] if 330 < x['bpm'] else 0)['power']
    else:
        bpm_first_peak = 0
        bpm_dominant_peak = 0
        power_first_peak = 0
        power_dominant_peak = 0
        max_power_under_110 = 0
        max_power_under_220 = 0
        max_power_under_330 = 0
        max_power_over_330 = 0

    # 高周波成分の解析
    is_high_power_extreme = 1 if np.any((f_sel >= high_freq_thr) & (P_sel >= power_thr)) else 0
    high_power_extremes = sum(1 for peak in FFT_peaks_powers 
                             if peak['freq'] >= high_freq_thr and peak['power'] >= power_thr) if is_high_power_extreme == 1 else 0

    # 高周波のpowerの和を計算
    high_freq_power_sum = sum(peak['power'] for peak in FFT_peaks_powers if peak['freq'] >= high_freq_thr)
    # 高周波のピーク数をカウント
    high_freq_peaks = sum(1 for peak in FFT_peaks_powers if peak['freq'] >= high_freq_thr) if len(FFT_peaks_powers) > 0 else 0
    
    # 高周波成分の面積を計算
    f_P = np.vstack((f_sel, P_sel)).T
    f_P_high = f_P[f_P[:, 0] >= high_freq_thr]
    mean_power_high_freq = np.trapezoid(f_P_high[:, 1], f_P_high[:, 0]) if len(f_P_high) > 0 else 0
    
    # スペクトル重心の計算
    bpm_spectrum_centroid = (np.sum(f_sel * P_sel) / np.sum(P_sel)) * 60

    # 特徴量辞書の作成
    features = {
        'bpm_first_peak': bpm_first_peak,  # 1st Peak(bpm)
        'bpm_dominant_peak': bpm_dominant_peak,  # パワー最大のピーク(bpm)
        'power_first_peak': power_first_peak,  # 1st Peak(power)
        'power_dominant_peak': power_dominant_peak,  # パワー最大のピーク(power)
        'max_power_under_110': max_power_under_110,  # 110bpm以下の最大パワー
        'max_power_under_220': max_power_under_220,  # 110~220bpmの最大パワー
        'max_power_under_330': max_power_under_330,  # 220~330bpmの最大パワー
        'max_power_over_330': max_power_over_330,  # 330bpm以上の最大パワー
        'is_high_power_extreme': is_high_power_extreme,  # 高周波パワーの有無
        'high_power_extremes': high_power_extremes,  # 高周波パワーの閾値以上のピーク数の合計
        'high_freq_power_sum': high_freq_power_sum,  # 高周波パワーの和
        'high_freq_peaks': high_freq_peaks,  # 高周波ピーク数
        'mean_power_high_freq': mean_power_high_freq,  # 高周波成分の面積
        'bpm_spectrum_centroid': bpm_spectrum_centroid  # スペクトル重心
    }
    
    # 可視化
    if plot:
        fig, ax = plt.subplots(2, figsize=(12, 6))
        
        # 上段：パワースペクトル
        ax[0].plot(f_sel, P_sel, label='Power Spectrum')
        ax[0].axvline(high_freq_thr, color='orange', linestyle='--', label=f'{high_freq_thr} Hz')
        ax[0].axhline(power_thr, color='gray', linestyle=':', label=f'Power >= {power_thr}')
        ax[0].fill_between(f_sel, P_sel, where=(f_sel >= high_freq_thr) & (P_sel >= power_thr),
                        color='red', alpha=0.3, label='High freq power')
        
        if bpm_first_peak:
            ax[0].axvline(bpm_first_peak / 60, color='green', linestyle='-', linewidth=2, 
                         label=f'1st Peak: {bpm_first_peak:.1f} bpm')
        if bpm_dominant_peak:
            ax[0].axvline(bpm_dominant_peak / 60, color='red', linestyle='--', linewidth=1, 
                         label=f'Dominant HR: {bpm_dominant_peak:.1f} bpm')
        if bpm_spectrum_centroid:
            ax[0].axvline(bpm_spectrum_centroid / 60, color='blue', linestyle='--', linewidth=1, 
                         label=f'Spectrum Centroid: {bpm_spectrum_centroid:.1f} bpm')
        
        if len(peaks) > 0:
            ax[0].scatter(f_sel[peaks], P_sel[peaks], color='red', label='Peaks')
        
        ax[0].fill_between(f_sel, P_sel, where=(f_sel >= high_freq_thr), 
                          color='orange', alpha=0.3, label='High freq power')
        ax[0].set_xlabel('Frequency (Hz)')
        ax[0].set_ylabel('Power')
        ax[0].set_xlim(0, 8)
        ax[0].grid(True)
        ax[0].legend(loc='upper right')
        ax[0].set_title(f'{file_name}  音量: {vol_mean:.0f}  \n HR: {actual_hr}, first: {bpm_first_peak}, dominant: {bpm_dominant_peak}')
        
        # 下段：時系列データ
        time_fft = np.arange(0, len(signal))/fs
        ax[1].plot(time_fft, signal, label='time series data')
        ax[1].set_xlabel('Time (s)')
        ax[1].set_ylabel('Amplitude')
        ax[1].set_xlim(0, len(time_fft)/fs)
        ax[1].grid(True)
        ax[1].legend(loc='upper right')
        
        plt.tight_layout()
        plt.show()
    
    return features