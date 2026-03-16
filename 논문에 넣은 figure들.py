import torch
import torch.nn as nn
import numpy as np
import joblib
import math
import sys
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.mplot3d import Axes3D

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_LEN, WINDOW_SIZE = 20, 20
INPUT_DIM, OUTPUT_DIM = 25, 3

# ==============================================================================
# 1. 모델 아키텍처 정의 (Transformer, LSTM, MLP, 1D-CNN)
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
# 2. 물리 및 RMSE 계산 함수
# ==============================================================================
r_cm, L_cm = 3.3, 7.9
def get_sensors_cm():
    S2 = np.sqrt(2)
    return np.array([[r_cm, 0, 0], [r_cm/S2, r_cm/S2, -L_cm], [0, r_cm, 0], [-r_cm/S2, r_cm/S2, -L_cm],
                     [-r_cm, 0, 0], [-r_cm/S2, -r_cm/S2, -L_cm], [0, -r_cm, 0], [r_cm/S2, -r_cm/S2, -L_cm]], dtype=np.float32)

def generate_controlled_traj_cm(td_noise_cm, doa_noise_deg, target_dist_cm=None):
    SOUND_SPEED_CM_S = 150000.0; sensors = get_sensors_cm(); traj = np.zeros((200, 3), dtype=np.float32)
    direction = np.random.randn(3); direction /= (np.linalg.norm(direction) + 1e-9)
    traj[0] = direction * target_dist_cm
    vec = np.random.randn(3); vec /= (np.linalg.norm(vec) + 1e-9)
    for i in range(1, 200):
        rv = np.random.randn(3); rv /= (np.linalg.norm(rv) + 1e-9)
        vec = 0.8 * vec + 0.2 * rv; vec /= (np.linalg.norm(vec) + 1e-9); traj[i] = traj[i-1] + vec * 100.0
    feats = np.zeros((200, 25), dtype=np.float32)
    td_std, doa_std = td_noise_cm / SOUND_SPEED_CM_S, np.radians(doa_noise_deg)
    for i, p in enumerate(traj):
        d = np.linalg.norm(sensors - p, axis=1); toa = (d / SOUND_SPEED_CM_S) + np.random.normal(0, td_std, size=8)
        dp = p - sensors
        feats[i] = np.concatenate([toa[0:1]*SOUND_SPEED_CM_S, (toa-toa[0])*SOUND_SPEED_CM_S, 
                                   np.arctan2(dp[:,1], dp[:,0]) + np.random.normal(0, doa_std, 8),
                                   np.arctan2(dp[:,2], np.sqrt(dp[:,0]**2 + dp[:,1]**2)+1e-9) + np.random.normal(0, doa_std, 8)])
    return traj, feats

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
# 3. 메인 분석부
# ==============================================================================
if __name__ == '__main__':
    CONFIG = {
        'proposed_path': 'model_td_0.0-5.0_doa_0.0-0.5_800m.pt',
        'lstm_path': 'model_lstm_800m.pt',
        'mlp_path': 'model_mlp_800m.pt',
        'cnn_path': 'model_cnn_800m.pt',
        'scaler_x': 'scaler_x_td_0.0-5.0_doa_0.0-0.5_800m.pkl',
        'scaler_y': 'scaler_y_td_0.0-5.0_doa_0.0-0.5_800m.pkl'
    }
    
    try:
        sx, sy = joblib.load(CONFIG['scaler_x']), joblib.load(CONFIG['scaler_y'])
        proposed_model = TransformerEncoderOnlyModel(25, 3, 128, 8, 10).to(DEVICE)
        proposed_model.load_state_dict(torch.load(CONFIG['proposed_path'], map_location=DEVICE))
        lstm_model = LSTMModel(25, 3, 256, 3, 0.3).to(DEVICE); lstm_model.load_state_dict(torch.load(CONFIG['lstm_path'], map_location=DEVICE))
        mlp_model = MLPModel(25, 3, 20, 512, dropout=0.3).to(DEVICE); mlp_model.load_state_dict(torch.load(CONFIG['mlp_path'], map_location=DEVICE))
        cnn_model = CNN1DModel(25, 3, dropout=0.3).to(DEVICE); cnn_model.load_state_dict(torch.load(CONFIG['cnn_path'], map_location=DEVICE))
    except FileNotFoundError: print("모델 파일 경로를 확인해주세요."); sys.exit()

    ITER = 100; 
    sensors_loc_cm = get_sensors_cm()

    def run_full_comparison(steps, type='dist'):
        res = {k: [] for k in ['Proposed', 'LSTM', 'MLP', 'KF', 'CNN']}
        for i, val in enumerate(steps):
            errs_cm = {k: [] for k in res.keys()}
            t_dist = val if type == 'dist' else 40000
            t_td = val if type == 'tdoa' else 2.5
            t_doa = val if type == 'doa' else 0.25
            for _ in range(ITER):
                gt_cm, feat_cm = generate_controlled_traj_cm(t_td, t_doa, target_dist_cm=t_dist)
                errs_cm['Proposed'].append(calculate_rmse(gt_cm, sliding_window_inference_cm(proposed_model, sx, sy, feat_cm)))
                errs_cm['LSTM'].append(calculate_rmse(gt_cm, sliding_window_inference_cm(lstm_model, sx, sy, feat_cm)))
                errs_cm['MLP'].append(calculate_rmse(gt_cm, sliding_window_inference_cm(mlp_model, sx, sy, feat_cm)))
                errs_cm['CNN'].append(calculate_rmse(gt_cm, sliding_window_inference_cm(cnn_model, sx, sy, feat_cm)))
                
                p_obs_cm = sliding_window_inference_cm(mlp_model, sx, sy, feat_cm)
                kf_t_cm, kf = [], KalmanFilter(gt_cm[0])
                for t in range(200): kf_t_cm.append(kf.predict_and_update(p_obs_cm[t]))
                errs_cm['KF'].append(calculate_rmse(gt_cm, np.array(kf_t_cm)))
                
            for k in res.keys(): res[k].append(np.mean(errs_cm[k]) / 100.0)
            sys.stdout.write(f'\r{type} 분석 중... ({i+1}/{len(steps)})'); sys.stdout.flush()
        return res

    dist_steps = np.linspace(0, 60000, 61)
    tdoa_steps = np.arange(0, 7.1, 0.1)
    doa_steps = np.arange(0, 1.3, 0.1)

    res_dist = run_full_comparison(dist_steps, 'dist')
    res_tdoa = run_full_comparison(tdoa_steps, 'tdoa')
    res_doa = run_full_comparison(doa_steps, 'doa')

    # --- 터미널 결과 출력 (순서 조정: Proposed -> LSTM -> MLP -> KF -> CNN) ---
    print(f"\n\n{'='*135}\n [ 600m 거리별 RMSE 요약 (m) ]\n{'='*135}")
    print(f"{'거리(m)':<10} | {'Proposed':<12} | {'LSTM':<12} | {'MLP':<12} | {'KF':<12} | {'1D-CNN'}")
    print(f"{'-'*135}")
    for i, d_cm in enumerate(dist_steps):
        if int(round(d_cm/100.0)) % 10 == 0:
            print(f"{int(round(d_cm/100.0)):>8} | {res_dist['Proposed'][i]:<12.4f} | {res_dist['LSTM'][i]:<12.4f} | {res_dist['MLP'][i]:<12.4f} | {res_dist['KF'][i]:<12.4f} | {res_dist['CNN'][i]:.4f}")

    print(f"\n{'='*135}\n [ TDOA 노이즈별 RMSE 요약 (m, 0.1cm 간격) ]\n{'='*135}")
    print(f"{'TDOA(cm)':<10} | {'Proposed':<12} | {'LSTM':<12} | {'MLP':<12} | {'KF':<12} | {'1D-CNN'}")
    print(f"{'-'*135}")
    for i in range(len(tdoa_steps)):
        print(f"{tdoa_steps[i]:>8.1f} | {res_tdoa['Proposed'][i]:<12.4f} | {res_tdoa['LSTM'][i]:<12.4f} | {res_tdoa['MLP'][i]:<12.4f} | {res_tdoa['KF'][i]:<12.4f} | {res_tdoa['CNN'][i]:.4f}")

    print(f"\n{'='*135}\n [ DOA 노이즈별 RMSE 요약 (m, 0.1deg 간격) ]\n{'='*135}")
    print(f"{'DOA(deg)':<10} | {'Proposed':<12} | {'LSTM':<12} | {'MLP':<12} | {'KF':<12} | {'1D-CNN'}")
    print(f"{'-'*135}")
    for i in range(len(doa_steps)):
        print(f"{doa_steps[i]:>8.1f} | {res_doa['Proposed'][i]:<12.4f} | {res_doa['LSTM'][i]:<12.4f} | {res_doa['MLP'][i]:<12.4f} | {res_doa['KF'][i]:<12.4f} | {res_doa['CNN'][i]:.4f}")

    # --- 시각화 데이터 생성 ---
    gt_cm, feat_cm = generate_controlled_traj_cm(2.5, 0.25, target_dist_cm=40000)
    p_prop_m = sliding_window_inference_cm(proposed_model, sx, sy, feat_cm) / 100.0
    p_cnn_m = sliding_window_inference_cm(cnn_model, sx, sy, feat_cm) / 100.0
    p_lstm_m = sliding_window_inference_cm(lstm_model, sx, sy, feat_cm) / 100.0
    p_mlp_m = sliding_window_inference_cm(mlp_model, sx, sy, feat_cm) / 100.0
    gt_m = gt_cm / 100.0
    
    kf_m_list, kf_vis = [], KalmanFilter(gt_m[0]*100.0)
    for t in range(200): kf_m_list.append(kf_vis.predict_and_update(p_mlp_m[t]*100.0)/100.0)
    p_kf_m = np.array(kf_m_list)

    # Figure 1-3: 평면별 추정도
    planes = [('X', 'Y', [0, 1], 1), ('X', 'Z', [0, 2], 2), ('Y', 'Z', [1, 2], 3)]
    for n1, n2, dims, fig_n in planes:
        plt.figure(fig_n, figsize=(8, 6))
        rms_p, rms_c, rms_l, rms_m, rms_k = [calculate_rmse(gt_m, p, dims) for p in [p_prop_m, p_cnn_m, p_lstm_m, p_mlp_m, p_kf_m]]
        plt.plot(gt_m[:, dims[0]], gt_m[:, dims[1]], 'k--', label='Ground Truth', lw=2)
        plt.plot(p_prop_m[:, dims[0]], p_prop_m[:, dims[1]], 'r-', label=f'Proposed (RMSE: {rms_p:.4f}m)', lw=1.5)
        plt.plot(p_lstm_m[:, dims[0]], p_lstm_m[:, dims[1]], 'm-', label=f'LSTM (RMSE: {rms_l:.4f}m)', lw=1.2, alpha=0.8)
        plt.plot(p_mlp_m[:, dims[0]], p_mlp_m[:, dims[1]], 'b-', label=f'MLP (RMSE: {rms_m:.4f}m)', lw=1.2, alpha=0.8)
        plt.plot(p_kf_m[:, dims[0]], p_kf_m[:, dims[1]], 'g-.', label=f'KF (RMSE: {rms_k:.4f}m)', lw=1.2, alpha=0.8)
        plt.plot(p_cnn_m[:, dims[0]], p_cnn_m[:, dims[1]], 'c-', label=f'1D-CNN (RMSE: {rms_c:.4f}m)', lw=1.2)
        plt.title(f'{n1}-{n2} 평면 추정도 (m)'); plt.xlabel(f'{n1} (m)'); plt.ylabel(f'{n2} (m)')
        plt.grid(True, ls=':', alpha=0.6); plt.legend(); plt.tight_layout()

    # Figure 4-6: 분석 그래프 (Log Scale)
    configs = [(4, dist_steps/100.0, res_dist, "Distance (m)", "거리별"),
               (5, tdoa_steps, res_tdoa, "TDOA Error (cm)", "TDOA 오차별"),
               (6, doa_steps, res_doa, "DOA Error (deg)", "DOA 오차별")]
    for fn, steps, res, xl, tit in configs:
        plt.figure(fn, figsize=(10, 7)); plt.plot(steps, res['Proposed'], 'r-', lw=2.5, label='Proposed')
        plt.plot(steps, res['LSTM'], 'm-', lw=1.2, label='LSTM', alpha=0.7)
        plt.plot(steps, res['MLP'], 'b-', lw=1.2, label='MLP', alpha=0.7)
        plt.plot(steps, res['KF'], 'g-.', lw=1.2, label='KF', alpha=0.8)
        plt.plot(steps, res['CNN'], 'c-', lw=1.5, label='1D-CNN')
        plt.yscale('log'); plt.ylim(0.1, 100); plt.gca().yaxis.set_major_formatter(ScalarFormatter())
        plt.title(tit); plt.xlabel(xl); plt.ylabel("RMSE (m)")
        plt.grid(True, which="both", ls=':', alpha=0.5); plt.legend(); plt.tight_layout()

    # Figure 7: 3D 궤적 시각화
    fig7 = plt.figure(7, figsize=(10, 8)); ax7 = fig7.add_subplot(111, projection='3d')
    rms_p3, rms_c3, rms_l3, rms_m3, rms_k3 = [calculate_rmse(gt_m, p) for p in [p_prop_m, p_cnn_m, p_lstm_m, p_mlp_m, p_kf_m]]
    sensors_m = sensors_loc_cm / 100.0
    ax7.scatter(sensors_m[:, 0], sensors_m[:, 1], sensors_m[:, 2], color='gold', s=100, label='Array Sensors', marker='^')
    ax7.plot(gt_m[:, 0], gt_m[:, 1], gt_m[:, 2], 'k--', label='Ground Truth', lw=2)
    ax7.plot(p_prop_m[:, 0], p_prop_m[:, 1], p_prop_m[:, 2], 'r-', label=f'Proposed (3D RMSE: {rms_p3:.4f}m)', lw=2)
    ax7.plot(p_lstm_m[:, 0], p_lstm_m[:, 1], p_lstm_m[:, 2], 'm-', label=f'LSTM (3D RMSE: {rms_l3:.4f}m)', lw=1.2, alpha=0.7)
    ax7.plot(p_mlp_m[:, 0], p_mlp_m[:, 1], p_mlp_m[:, 2], 'b-', label=f'MLP (3D RMSE: {rms_m3:.4f}m)', lw=1.2, alpha=0.7)
    ax7.plot(p_kf_m[:, 0], p_kf_m[:, 1], p_kf_m[:, 2], 'g-.', label=f'KF (3D RMSE: {rms_k3:.4f}m)', lw=1.2, alpha=0.7)
    ax7.plot(p_cnn_m[:, 0], p_cnn_m[:, 1], p_cnn_m[:, 2], 'c-', label=f'1D-CNN (3D RMSE: {rms_c3:.4f}m)', lw=1.2)
    ax7.set_title('3D Trajectory Localization (m)'); ax7.legend(); plt.show()