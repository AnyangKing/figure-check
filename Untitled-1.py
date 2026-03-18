import torch
import torch.nn as nn
import numpy as np
import joblib
import math
import sys
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, MultipleLocator
from mpl_toolkits.mplot3d import Axes3D

# Matplotlib 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_LEN, WINDOW_SIZE = 20, 20
INPUT_DIM, OUTPUT_DIM = 25, 3
SOUND_SPEED_CM_S = 150000.0

# ==============================================================================
# 1. 모델 아키텍처 정의 (유지)
# ==============================================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=20):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term); pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0); self.register_buffer('pe', pe)
    def forward(self, x): return self.dropout(x + self.pe[:, :x.size(1), :])

class TransformerEncoderOnlyModel(nn.Module):
    def __init__(self, input_dim, output_dim, d_model, nhead, nlayers, dropout=0.0528):
        super(TransformerEncoderOnlyModel, self).__init__()
        self.d_model = d_model
        self.encoder_embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=MAX_LEN)
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=nlayers)
        self.fc_out = nn.Linear(d_model, output_dim)
    def forward(self, src):
        src = self.encoder_embedding(src) * math.sqrt(self.d_model)
        return self.fc_out(self.transformer_encoder(self.pos_encoder(src)))

class LSTMModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers, dropout):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        out, _ = self.lstm(x); return self.fc(out)

class MLPModel(nn.Module):
    def __init__(self, input_dim, output_dim, window_size, hidden_dim, dropout=0.3):
        super(MLPModel, self).__init__()
        self.window_size = window_size
        self.net = nn.Sequential(
            nn.Linear(window_size * input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, window_size * output_dim)
        )
    def forward(self, x):
        batch_size = x.size(0); x = x.view(batch_size, -1); out = self.net(x)
        return out.view(batch_size, self.window_size, -1)

class CNN1DModel(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.3):
        super(CNN1DModel, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_dim, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.GELU()
        )
        self.output_layer = nn.Conv1d(128, output_dim, kernel_size=1)
    def forward(self, x):
        x = x.transpose(1, 2)
        out = self.output_layer(self.conv_layers(x))
        return out.transpose(1, 2)

class KalmanFilter:
    def __init__(self, init_pos):
        self.dt = 1.0; self.x = np.array([init_pos[0], init_pos[1], init_pos[2], 0, 0, 0])
        self.F = np.eye(6); self.F[0, 3], self.F[1, 4], self.F[2, 5] = 1, 1, 1
        self.H = np.zeros((3, 6)); self.H[0, 0], self.H[1, 1], self.H[2, 2] = 1, 1, 1
        self.P, self.Q, self.R = np.eye(6)*500.0, np.eye(6)*1.0, np.eye(3)*100.0
    def predict_and_update(self, z):
        self.x = self.F @ self.x; self.P = self.F @ self.P @ self.F.T + self.Q
        S = self.H @ self.P @ self.H.T + self.R; K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - self.H @ self.x); self.P = (np.eye(6) - K @ self.H) @ self.P
        return self.x[:3]

# ==============================================================================
# 2. 물리 및 데이터 생성 (유지)
# ==============================================================================
r_cm, L_cm = 3.3, 7.9
def get_sensors_cm():
    S2 = np.sqrt(2)
    return np.array([[r_cm, 0, 0], [r_cm/S2, r_cm/S2, -L_cm], [0, r_cm, 0], [-r_cm/S2, r_cm/S2, -L_cm],
                     [-r_cm, 0, 0], [-r_cm/S2, -r_cm/S2, -L_cm], [0, -r_cm, 0], [r_cm/S2, -r_cm/S2, -L_cm]], dtype=np.float32)

def generate_controlled_traj_cm(td_noise_cm, doa_noise_deg, target_dist_cm=None, max_bias_cm=0.0):
    sensors = get_sensors_cm(); traj = np.zeros((200, 3), dtype=np.float32)
    direction = np.random.randn(3); direction /= (np.linalg.norm(direction) + 1e-9)
    traj[0] = direction * target_dist_cm
    vec = np.random.randn(3); vec /= (np.linalg.norm(vec) + 1e-9)
    for i in range(1, 200):
        rv = np.random.randn(3); rv /= (np.linalg.norm(rv) + 1e-9)
        vec = 0.8 * vec + 0.2 * rv; vec /= (np.linalg.norm(vec) + 1e-9); traj[i] = traj[i-1] + vec * 100.0
    
    feats = np.zeros((200, 25), dtype=np.float32)
    td_std, doa_std = td_noise_cm / SOUND_SPEED_CM_S, np.radians(doa_noise_deg)
    sensor_specific_biases = np.random.uniform(-max_bias_cm, max_bias_cm, size=8) / SOUND_SPEED_CM_S
    for i, p in enumerate(traj):
        d = np.linalg.norm(sensors - p, axis=1)
        toa = (d / SOUND_SPEED_CM_S) + sensor_specific_biases + np.random.normal(0, td_std, size=8)
        dp = p - sensors
        feats[i] = np.concatenate([toa[0:1]*SOUND_SPEED_CM_S, (toa-toa[0])*SOUND_SPEED_CM_S, 
                                   np.arctan2(dp[:,1], dp[:,0]) + np.random.normal(0, doa_std, 8),
                                   np.arctan2(dp[:,2], np.sqrt(dp[:,0]**2 + dp[:,1]**2)+1e-9) + np.random.normal(0, doa_std, 8)])
    return traj, feats

def music_doa_estimation_stable(sensors, target_pos, doa_error_deg):
    array_center = np.mean(sensors, axis=0)
    true_vec = (target_pos - array_center); true_vec /= (np.linalg.norm(true_vec) + 1e-9)
    noise_std = np.tan(np.radians(doa_error_deg))
    noise = np.random.normal(0, noise_std + 1e-12, 3)
    est_vec = true_vec + noise
    return est_vec / (np.linalg.norm(est_vec) + 1e-9)

def localize_music(sensors, estimated_doa, true_dist_cm, td_noise_cm, max_bias_cm=0.0):
    bias_val = np.random.uniform(-max_bias_cm, max_bias_cm)
    noisy_dist = true_dist_cm + bias_val + np.random.normal(0, td_noise_cm)
    return np.mean(sensors, axis=0) + (estimated_doa * noisy_dist)

def sliding_window_inference_cm(model, sx, sy, x_raw_cm):
    model.eval(); x_scaled = sx.transform(x_raw_cm)
    windows = torch.FloatTensor(np.array([x_scaled[i:i+WINDOW_SIZE, :] for i in range(200 - WINDOW_SIZE + 1)])).to(DEVICE)
    final, counts = np.zeros((200, 3)), np.zeros((200, 1))
    with torch.no_grad():
        preds = model(windows).cpu().numpy()
        for i in range(len(preds)): final[i:i+WINDOW_SIZE, :] += preds[i]; counts[i:i+WINDOW_SIZE, :] += 1
    return sy.inverse_transform(final / (counts + 1e-9))

def calculate_rmse(gt, pred, dims=[0, 1, 2]):
    return np.sqrt(np.mean(np.sum((gt[:, dims] - pred[:, dims])**2, axis=1)))

# ==============================================================================
# 3. 메인 분석부 (터미널 표 복원 및 그리드 최적화)
# ==============================================================================
if __name__ == '__main__':
    model_styles = {
        'Proposed': {'marker': 'o', 'color': 'r', 'ls': '-'},
        'MUSIC':    {'marker': 'D', 'color': 'green', 'ls': '--'},
        'LSTM':     {'marker': 's', 'color': 'm', 'ls': '-'},
        'MLP':      {'marker': '^', 'color': 'b', 'ls': '-'},
        'KF':       {'marker': 'P', 'color': 'orange', 'ls': '-'}, 
        'CNN':      {'marker': 'x', 'color': 'c', 'ls': '-'}
    }
    LINE_WIDTH = 1.5
    CONFIG = {
        'proposed_path': 'model_td_0.0-5.0_doa_0.0-0.5_800m.pt', 'lstm_path': 'model_lstm_800m.pt',
        'mlp_path': 'model_mlp_800m.pt', 'cnn_path': 'model_cnn_800m.pt',
        'scaler_x': 'scaler_x_td_0.0-5.0_doa_0.0-0.5_800m.pkl', 'scaler_y': 'scaler_y_td_0.0-5.0_doa_0.0-0.5_800m.pkl'
    }
    try:
        sx, sy = joblib.load(CONFIG['scaler_x']), joblib.load(CONFIG['scaler_y'])
        proposed_model = TransformerEncoderOnlyModel(25, 3, 128, 8, 10).to(DEVICE); proposed_model.load_state_dict(torch.load(CONFIG['proposed_path'], map_location=DEVICE))
        lstm_model = LSTMModel(25, 3, 256, 3, 0.3).to(DEVICE); lstm_model.load_state_dict(torch.load(CONFIG['lstm_path'], map_location=DEVICE))
        mlp_model = MLPModel(25, 3, 20, 512, dropout=0.3).to(DEVICE); mlp_model.load_state_dict(torch.load(CONFIG['mlp_path'], map_location=DEVICE))
        cnn_model = CNN1DModel(25, 3, dropout=0.3).to(DEVICE); cnn_model.load_state_dict(torch.load(CONFIG['cnn_path'], map_location=DEVICE))
    except FileNotFoundError: print("모델 파일 경로를 확인해주세요."); sys.exit()

    ITER = 1000 
    sensors_loc_cm = get_sensors_cm()

    def run_full_comparison(steps, type='dist'):
        res = {k: [] for k in model_styles.keys()}
        CONTROL_DIST_CM, CONTROL_TDOA_STD_CM, CONTROL_DOA_DEG = 40000, 7.5, 0.5 
        for i, val in enumerate(steps):
            errs_cm = {k: [] for k in res.keys()}
            t_dist, t_td_std = (val if type == 'dist' else CONTROL_DIST_CM), CONTROL_TDOA_STD_CM
            t_td_bias, t_doa = (val if type == 'tdoa' else 0.0), (val if type == 'doa' else CONTROL_DOA_DEG)
            for _ in range(ITER):
                gt_cm, feat_cm = generate_controlled_traj_cm(t_td_std, t_doa, target_dist_cm=t_dist, max_bias_cm=t_td_bias)
                for k, m in zip(['Proposed', 'LSTM', 'MLP', 'CNN'], [proposed_model, lstm_model, mlp_model, cnn_model]):
                    errs_cm[k].append(calculate_rmse(gt_cm, sliding_window_inference_cm(m, sx, sy, feat_cm)))
                p_obs_cm = sliding_window_inference_cm(mlp_model, sx, sy, feat_cm)
                kf_t_cm, kf = [], KalmanFilter(gt_cm[0])
                for t in range(200): kf_t_cm.append(kf.predict_and_update(p_obs_cm[t]))
                errs_cm['KF'].append(calculate_rmse(gt_cm, np.array(kf_t_cm)))
                p_music_traj = []
                for t in range(200):
                    dist_to_c = np.linalg.norm(gt_cm[t] - np.mean(sensors_loc_cm, axis=0))
                    m_doa = music_doa_estimation_stable(sensors_loc_cm, gt_cm[t], t_doa)
                    p_music_traj.append(localize_music(sensors_loc_cm, m_doa, dist_to_c, t_td_std, max_bias_cm=t_td_bias))
                errs_cm['MUSIC'].append(calculate_rmse(gt_cm, np.array(p_music_traj)))
            for k in res.keys(): res[k].append(np.mean(errs_cm[k]) / 100.0)
            sys.stdout.write(f'\r{type} 분석 중... ({i+1}/{len(steps)})'); sys.stdout.flush()
        return res

    dist_steps = np.linspace(0, 60000, 61) 
    tdoa_bias_steps_cm = np.linspace(0, 15, 11) 
    doa_steps = np.linspace(0, 1.2, 13) 

    res_dist = run_full_comparison(dist_steps, 'dist')
    res_tdoa = run_full_comparison(tdoa_bias_steps_cm, 'tdoa')
    res_doa = run_full_comparison(doa_steps, 'doa')

    # --- 터미널 결과 출력 (지적 사항 반영 유지) ---
    for title, steps, data, label in [("거리별 (m)", dist_steps, res_dist, "Dist(m)"), 
                                      ("TDOA 개별 바이어스별 (us)", tdoa_bias_steps_cm, res_tdoa, "Bias(us)"), 
                                      ("DOA 검증별 (deg)", doa_steps, res_doa, "DOA(deg)")]:
        print(f"\n\n{'='*175}\n [ 종합 RMSE 비교 요약: {title} ]\n{'='*175}")
        if "DOA" in title: header = f"{label:<10} | {'Prop':<16} | {'LSTM':<16} | {'MLP':<16} | {'KF':<16} | {'1D-CNN'}"
        else: header = f"{label:<10} | {'Prop':<16} | {'MUSIC':<16} | {'LSTM':<16} | {'MLP':<16} | {'KF':<16} | {'1D-CNN'}"
        print(header); print(f"{'-'*175}")
        for i in range(len(steps)):
            if "거리" in title:
                val = int(round(steps[i]/100.0))
                if val % 10 != 0: continue
                step_str = f"{val:>8}"
            elif "Bias" in title: step_str = f"{round(i*10.0):>8}"
            else: step_str = f"{steps[i]:>8.1f}"
            row = f"{step_str} | {data['Proposed'][i]:<16.4f} | "
            if "DOA" not in title: row += f"{data['MUSIC'][i]:<16.4f} | " 
            row += f"{data['LSTM'][i]:<16.4f} | {data['MLP'][i]:<16.4f} | {data['KF'][i]:<16.4f} | {data['CNN'][i]:.4f}"
            print(row)

    # --- 시각화 데이터 생성 ---
    gt_cm, feat_cm = generate_controlled_traj_cm(7.5, 0.5, target_dist_cm=40000, max_bias_cm=7.5)
    p_all = {k: sliding_window_inference_cm(m, sx, sy, feat_cm)/100.0 for k, m in zip(['Proposed', 'CNN', 'LSTM', 'MLP'], [proposed_model, cnn_model, lstm_model, mlp_model])}
    gt_m, music_m_list, kf_vis = gt_cm/100.0, [], KalmanFilter(gt_cm[0])
    for t in range(200):
        m_v = music_doa_estimation_stable(sensors_loc_cm, gt_cm[t], 0.5)
        music_m_list.append(localize_music(sensors_loc_cm, m_v, np.linalg.norm(gt_cm[t]-np.mean(sensors_loc_cm,0)), 7.5, 7.5)/100.0)
    p_all['MUSIC'], p_all['KF'] = np.array(music_m_list), np.array([kf_vis.predict_and_update(p_all['MLP'][t]*100.0)/100.0 for t in range(200)])

    # [수정] Figure 4: 전체 거리별 분석 (세로선 100m 단위 최적화)
    plt.figure(4, figsize=(10, 7)); steps_m = dist_steps/100.0
    plt.gca().set_xticks(np.arange(0, 601, 100)) # [핵심] 세로선을 100m 단위로만 설정
    plt.gca().xaxis.grid(True, ls=':', alpha=0.5); plt.gca().yaxis.grid(True, which='both', ls=':', alpha=0.5)
    for k in model_styles.keys():
        # [핵심] 마커를 세로선(100m) 마다 표시 (markevery=10, 데이터가 10m 단위이므로)
        plt.plot(steps_m, res_dist[k], label=('1D-CNN' if k=='CNN' else k), color=model_styles[k]['color'], marker=model_styles[k]['marker'], ls=model_styles[k]['ls'], lw=LINE_WIDTH, markevery=10, markersize=7)
    plt.yscale('log'); plt.ylim(0.1, 100); plt.gca().yaxis.set_major_formatter(ScalarFormatter()); plt.title("거리별 오차 분석"); plt.xlabel("Distance (m)"); plt.ylabel("RMSE (m)"); plt.legend(); plt.tight_layout()

    # Figure 5, 6 (바이어스/DOA - 유지)
    td_us = (tdoa_bias_steps_cm / SOUND_SPEED_CM_S) * 1000000
    for fn, steps, res_data, xl, tit in [(5, td_us, res_tdoa, r"TDOA Bias ($\mu s$)", "TDOA 개별 바이어스 검증 (Std=50us 고정)"), (6, doa_steps, res_doa, "DOA Deviation (deg)", "DOA Validation (환경 강인성 검증)")]:
        plt.figure(fn, figsize=(10, 7)); plt.gca().set_xticks(steps); plt.gca().xaxis.grid(True, ls=':', alpha=0.5); plt.gca().yaxis.grid(True, which='both', ls=':', alpha=0.5)
        for k in model_styles.keys():
            if fn == 6 and k == 'MUSIC': continue
            plt.plot(steps, res_data[k], label=('1D-CNN' if k=='CNN' else k), color=model_styles[k]['color'], marker=model_styles[k]['marker'], ls=model_styles[k]['ls'], lw=LINE_WIDTH, markevery=1, markersize=7)
        plt.yscale('log'); plt.ylim(0.1, 100); plt.gca().yaxis.set_major_formatter(ScalarFormatter()); plt.title(tit); plt.xlabel(xl); plt.ylabel("RMSE (m)"); plt.legend(); plt.tight_layout()

    # [수정] Figure 8: 100m ~ 300m 상세 (가로선 복원 및 세로선 20m 단위)
    plt.figure(8, figsize=(10, 7)); mask = (dist_steps >= 10000) & (dist_steps <= 30000); steps_sub = dist_steps[mask]/100.0
    plt.gca().set_xticks(np.arange(100, 301, 20)) # [핵심] 세로선을 20m 단위로 설정
    plt.gca().xaxis.grid(True, ls=':', alpha=0.5)
    plt.gca().yaxis.grid(True, which='major', ls=':', alpha=0.5) # [핵심] 사라진 가로선(그리드) 강제 활성화
    for k in model_styles.keys():
        # [핵심] 마커를 세로선(20m) 마다 표시 (markevery=2)
        plt.plot(steps_sub, np.array(res_dist[k])[mask], label=('1D-CNN' if k=='CNN' else k), color=model_styles[k]['color'], marker=model_styles[k]['marker'], ls=model_styles[k]['ls'], lw=2.0, markevery=2)
    plt.title("거리별 오차 분석 (100m ~ 300m 구간 상세)"); plt.xlabel("Distance (m)"); plt.ylabel("RMSE (m)"); plt.legend(); plt.tight_layout()

    # Figure 1-3, 7 (궤적 - 유지)
    planes = [('X', 'Y', [0, 1], 1), ('X', 'Z', [0, 2], 2), ('Y', 'Z', [1, 2], 3)]
    for n1, n2, dims, fig_n in planes:
        plt.figure(fig_n, figsize=(9, 7)); plt.plot(gt_m[:, dims[0]], gt_m[:, dims[1]], 'k--', label='Ground Truth', lw=2)
        for k in ['Proposed', 'MUSIC', 'LSTM', 'MLP', 'KF', 'CNN']: plt.plot(p_all[k][:, dims[0]], p_all[k][:, dims[1]], label=k, color=model_styles[k]['color'], marker=model_styles[k]['marker'], ls=model_styles[k]['ls'], lw=LINE_WIDTH, markevery=15)
        plt.title(f'{n1}-{n2} 평면 추기도 (m)'); plt.grid(True, ls=':', alpha=0.6); plt.legend(); plt.tight_layout()
    fig7 = plt.figure(7, figsize=(10, 8)); ax7 = fig7.add_subplot(111, projection='3d'); ax7.plot(gt_m[:, 0], gt_m[:, 1], gt_m[:, 2], 'k--', label='Ground Truth', lw=2)
    for k in ['Proposed', 'MUSIC', 'LSTM', 'MLP', 'KF', 'CNN']: ax7.plot(p_all[k][:, 0], p_all[k][:, 1], p_all[k][:, 2], label=k, color=model_styles[k]['color'], marker=model_styles[k]['marker'], ls=model_styles[k]['ls'], lw=LINE_WIDTH, markevery=20)
    ax7.set_title('3D Trajectory Localization (m)'); ax7.legend(); plt.tight_layout()

    plt.show()