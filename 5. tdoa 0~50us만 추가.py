import torch
import torch.nn as nn
import numpy as np
import joblib
import math
import sys
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# Matplotlib 한글 폰트 및 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
WINDOW_SIZE = 20
SOUND_SPEED_CM_S = 150000.0
ITER = 1000  # 사용자님 설정값: 10만 회

# ==============================================================================
# 1. 모델 아키텍처 (추론을 위해 필요한 최소한의 정의)
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
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=20)
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
    def forward(self, x): out, _ = self.lstm(x); return self.fc(out)

class MLPModel(nn.Module):
    def __init__(self, input_dim, output_dim, window_size, hidden_dim, dropout=0.3):
        super(MLPModel, self).__init__()
        self.window_size = window_size
        self.net = nn.Sequential(
            nn.Linear(window_size * input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim * 2), nn.BatchNorm1d(hidden_dim * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, window_size * output_dim)
        )
    def forward(self, x):
        bs = x.size(0); x = x.view(bs, -1); out = self.net(x); return out.view(bs, self.window_size, -1)

class CNN1DModel(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.3):
        super(CNN1DModel, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_dim, 128, 3, padding=1), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(128, 256, 3, padding=1), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(256, 128, 3, padding=1), nn.BatchNorm1d(128), nn.GELU()
        ); self.output_layer = nn.Conv1d(128, output_dim, 1)
    def forward(self, x): x = x.transpose(1, 2); out = self.output_layer(self.conv_layers(x)); return out.transpose(1, 2)

class KalmanFilter:
    def __init__(self, init_pos):
        self.x = np.array([init_pos[0], init_pos[1], init_pos[2], 0, 0, 0])
        self.F = np.eye(6); self.F[0,3]=self.F[1,4]=self.F[2,5]=1.0
        self.H = np.zeros((3, 6)); self.H[0,0]=self.H[1,1]=self.H[2,2]=1.0
        self.P, self.Q, self.R = np.eye(6)*500, np.eye(6)*1.0, np.eye(3)*100
    def predict_and_update(self, z):
        self.x = self.F @ self.x; self.P = self.F @ self.P @ self.F.T + self.Q
        S = self.H @ self.P @ self.H.T + self.R; K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - self.H @ self.x); self.P = (np.eye(6) - K @ self.H) @ self.P
        return self.x[:3]

# ==============================================================================
# 2. 도움 함수 (물리 엔진 및 추론)
# ==============================================================================
r_cm, L_cm = 3.3, 7.9
def get_sensors_cm():
    S2 = np.sqrt(2)
    return np.array([[r_cm, 0, 0], [r_cm/S2, r_cm/S2, -L_cm], [0, r_cm, 0], [-r_cm/S2, r_cm/S2, -L_cm],
                     [-r_cm, 0, 0], [-r_cm/S2, -r_cm/S2, -L_cm], [0, -r_cm, 0], [r_cm/S2, -r_cm/S2, -L_cm]], dtype=np.float32)

def generate_controlled_traj_cm(td_noise_cm, doa_noise_deg, target_dist_cm=None, m_bias_cm=0.0):
    sensors = get_sensors_cm(); traj = np.zeros((200, 3), dtype=np.float32)
    direction = np.random.randn(3); direction /= (np.linalg.norm(direction) + 1e-9)
    traj[0] = direction * target_dist_cm; vec = np.random.randn(3); vec /= (np.linalg.norm(vec) + 1e-9)
    for i in range(1, 200):
        rv = np.random.randn(3); rv /= (np.linalg.norm(rv) + 1e-9)
        vec = 0.8*vec + 0.2*rv; vec /= (np.linalg.norm(vec)+1e-9); traj[i] = traj[i-1] + vec*100.0
    
    feats = np.zeros((200, 25), dtype=np.float32)
    td_std, doa_std = td_noise_cm / SOUND_SPEED_CM_S, np.radians(doa_noise_deg)
    sensor_specific_biases = np.random.normal(m_bias_cm, m_bias_cm * 0.5 + 1e-9, size=8) / SOUND_SPEED_CM_S
    for i, p in enumerate(traj):
        d = np.linalg.norm(sensors - p, axis=1)
        toa = (d / SOUND_SPEED_CM_S) + sensor_specific_biases + np.random.normal(0, td_std, size=8)
        dp = p - sensors
        feats[i] = np.concatenate([toa[0:1]*SOUND_SPEED_CM_S, (toa-toa[0])*SOUND_SPEED_CM_S, 
                                   np.arctan2(dp[:,1], dp[:,0]) + np.random.normal(0, doa_std, 8),
                                   np.arctan2(dp[:,2], np.sqrt(dp[:,0]**2 + dp[:,1]**2)+1e-9) + np.random.normal(0, doa_std, 8)])
    return traj, feats

def music_doa_estimation_stable(sensors, target_pos, doa_error_deg):
    array_center = np.mean(sensors, axis=0); true_vec = (target_pos - array_center); true_vec /= (np.linalg.norm(true_vec) + 1e-9)
    noise = np.random.normal(0, np.tan(np.radians(doa_error_deg)) + 1e-12, 3)
    est_vec = true_vec + noise; return est_vec / (np.linalg.norm(est_vec) + 1e-9)

def localize_music(sensors, estimated_doa, true_dist_cm, td_noise_cm, m_bias_cm=0.0):
    bias_val = np.random.normal(m_bias_cm, m_bias_cm * 0.5 + 1e-9)
    return np.mean(sensors, axis=0) + (estimated_doa * (true_dist_cm + bias_val + np.random.normal(0, td_noise_cm)))

def sliding_window_inference_cm(model, sx, sy, x_raw_cm):
    model.eval(); x_scaled = sx.transform(x_raw_cm); windows = torch.FloatTensor(np.array([x_scaled[i:i+WINDOW_SIZE, :] for i in range(200 - WINDOW_SIZE + 1)])).to(DEVICE)
    final, counts = np.zeros((200, 3)), np.zeros((200, 1))
    with torch.no_grad():
        preds = model(windows).cpu().numpy()
        for i in range(len(preds)): final[i:i+WINDOW_SIZE, :] += preds[i]; counts[i:i+WINDOW_SIZE, :] += 1
    return sy.inverse_transform(final / (counts + 1e-9))

# ==============================================================================
# 3. 메인 실행 (TDOA 0~50us 전용)
# ==============================================================================
if __name__ == '__main__':
    # 모델 로드 (경로 주의)
    CONFIG = {
        'proposed_path': 'model_td_0.0-5.0_doa_0.0-0.5_800m.pt', 'lstm_path': 'model_lstm_800m.pt',
        'mlp_path': 'model_mlp_800m.pt', 'cnn_path': 'model_cnn_800m.pt',
        'scaler_x': 'scaler_x_td_0.0-5.0_doa_0.0-0.5_800m.pkl', 'scaler_y': 'scaler_y_td_0.0-5.0_doa_0.0-0.5_800m.pkl'
    }
    
    try:
        sx, sy = joblib.load(CONFIG['scaler_x']), joblib.load(CONFIG['scaler_y'])
        p_m = TransformerEncoderOnlyModel(25, 3, 128, 8, 10).to(DEVICE); p_m.load_state_dict(torch.load(CONFIG['proposed_path'], map_location=DEVICE))
        l_m = LSTMModel(25, 3, 256, 3, 0.3).to(DEVICE); l_m.load_state_dict(torch.load(CONFIG['lstm_path'], map_location=DEVICE))
        m_m = MLPModel(25, 3, 20, 512, 0.3).to(DEVICE); m_m.load_state_dict(torch.load(CONFIG['mlp_path'], map_location=DEVICE))
        c_m = CNN1DModel(25, 3, 0.3).to(DEVICE); c_m.load_state_dict(torch.load(CONFIG['cnn_path'], map_location=DEVICE))
    except Exception as e:
        print(f"파일 로드 실패: {e}"); sys.exit()

    # [수정] 0~50us 구간 설정 (1us 간격)
    tdoa_m_steps_cm = np.linspace(0, 7.5, 51)  # 50us = 7.5cm
    model_styles = {
        'Proposed': {'marker': 'o', 'color': 'r', 'ls': '-'},
        'MUSIC': {'marker': 'D', 'color': 'green', 'ls': '--'},
        'LSTM': {'marker': 's', 'color': 'm', 'ls': '-'},
        'MLP': {'marker': '^', 'color': 'b', 'ls': '-'},
        'KF': {'marker': 'P', 'color': 'orange', 'ls': '-'},
        'CNN': {'marker': 'x', 'color': 'c', 'ls': '-'}
    }
    
    results = {k: [] for k in model_styles.keys()}
    sensors_loc_cm = get_sensors_cm()

    print(f"--- TDOA 바이어스 분석 시작 (0-50us, ITER={ITER}) ---")
    for i, m_val in enumerate(tdoa_m_steps_cm):
        errs = {k: [] for k in results.keys()}
        for j in range(ITER):
            gt, feat = generate_controlled_traj_cm(7.5, 0.5, 40000, m_val) # 50us 기본 노이즈
            # 각 모델 추론
            for k, m in zip(['Proposed', 'LSTM', 'MLP', 'CNN'], [p_m, l_m, m_m, c_m]):
                pred = sliding_window_inference_cm(m, sx, sy, feat)
                errs[k].append(np.sqrt(np.mean(np.sum((gt - pred)**2, axis=1))))
            
            # KF (MLP 기반)
            p_o = sliding_window_inference_cm(m_m, sx, sy, feat); kf_t, kf = [], KalmanFilter(gt[0])
            for t in range(200): kf_t.append(kf.predict_and_update(p_o[t]))
            errs['KF'].append(np.sqrt(np.mean(np.sum((gt - np.array(kf_t))**2, axis=1))))
            
            # MUSIC
            p_mus = []
            for t in range(200):
                m_v = music_doa_estimation_stable(sensors_loc_cm, gt[t], 0.5)
                p_mus.append(localize_music(sensors_loc_cm, m_v, np.linalg.norm(gt[t]-np.mean(sensors_loc_cm,0)), 7.5, m_val))
            errs['MUSIC'].append(np.sqrt(np.mean(np.sum((gt - np.array(p_mus))**2, axis=1))))

            if (j+1) % (ITER//10) == 0:
                sys.stdout.write(f'\rPoint {i+1}/51 | Progress: {((j+1)/ITER)*100:.0f}%'); sys.stdout.flush()
        
        for k in results.keys(): results[k].append(np.mean(errs[k]) / 100.0)

    # ==============================================================================
    # 4. 시각화 (0~50us 전용)
    # ==============================================================================
    plt.figure(figsize=(10, 7))
    td_us = (tdoa_m_steps_cm / SOUND_SPEED_CM_S) * 1000000
    
    plt.gca().set_xticks(np.arange(0, 51, 10)) # 10us 단위 세로선
    plt.gca().xaxis.grid(True, ls=':', alpha=0.5)
    plt.gca().yaxis.grid(True, which='both', ls=':', alpha=0.5)
    
    for k in model_styles.keys():
        plt.plot(td_us, results[k], label=k, color=model_styles[k]['color'], 
                 marker=model_styles[k]['marker'], ls=model_styles[k]['ls'], lw=2.0, markevery=5)

    plt.yscale('log'); plt.ylim(0.1, 50)
    plt.gca().yaxis.set_major_formatter(ScalarFormatter())
    plt.title("TDOA 정밀 바이어스 분석 (0 ~ 50 $\mu$s 구간)")
    plt.xlabel(r"TDOA Bias ($\mu$s)")
    plt.ylabel("RMSE (m)")
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()