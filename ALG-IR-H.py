import numpy as np
from scipy.stats import truncnorm
from scipy.optimize import brentq

# ==============================================================================
# CLASS TÍNH TOÁN ALG-IR-H (CÓ PHÍ LƯU KHO)
# ==============================================================================
class HoldingCostSimulator:
    def __init__(self, Q, m, M, h, a, b, delta=0.2):
        self.Q_init = Q
        self.m = m
        self.M = M
        self.h = h 
        self.a = a
        self.b = b
        
        # Cấu hình phân phối
        a_trunc, b_trunc = (1 - delta - 1) / 1.0, (1 + delta - 1) / 1.0
        self.dist = truncnorm(a_trunc, b_trunc, loc=1, scale=1)

    def _F_inverse(self, val):
        val = max(1e-6, min(1-1e-6, val))
        return self.dist.ppf(val)
    
    def _F_cdf(self, val):
        return self.dist.cdf(val)

    def _get_dynamic_params(self, t_current):
        # Tính giá trị thực tế sau khi trừ phí lưu kho
        holding_deduction = (t_current - 1) * self.h
        effective_m = self.m - holding_deduction
        effective_M = self.M - holding_deduction
        
        if effective_m <= 0.1: return 1000, 0.1 
        theta_t = effective_M / effective_m
        return theta_t, effective_m

    def _phi_h(self, z, t_current):
        theta_t, eff_m = self._get_dynamic_params(t_current)
        threshold_t = self.Q_init / (1 + np.log(theta_t))
        
        if z <= threshold_t: return eff_m
        else:
            exponent = ((1 + np.log(theta_t)) / self.Q_init) * z - 1
            return eff_m * np.exp(exponent)

    def _pi_prime_h(self, x, p, t_current):
        holding_cost_accumulated = (t_current - 1) * self.h
        D_mean = self.a - self.b * p
        
        if x <= D_mean * 0.8: mr = p
        elif x >= D_mean * 1.2: mr = 0
        else: mr = p * (1 - self._F_cdf(x / D_mean))
        
        return mr - holding_cost_accumulated

    def decide_with_stage(self, p, current_y, t_current, is_last):
        """
        Trả về cả Quyết định (x) và Tên giai đoạn (stage_name)
        """
        # 1. Ngày cuối
        if is_last:
            return max(0, self.Q_init - current_y), "Xả kho"

        theta_t, _ = self._get_dynamic_params(t_current)
        threshold_t = self.Q_init / (1 + np.log(theta_t))

        # 2. Giai đoạn 1 (Stage 1)
        D_mean = self.a - self.b * p
        val_prob = 1 - (self.m / p) 
        
        if val_prob <= 0: x_trial = 0
        else: x_trial = D_mean * self._F_inverse(val_prob)

        if current_y + x_trial <= threshold_t:
            return max(0, x_trial), "Stage 1"
        
        # 3. Giai đoạn 2 (Stage 2)
        def equation(x):
            return self._pi_prime_h(x, p, t_current) - self._phi_h(current_y + x, t_current)

        remaining = self.Q_init - current_y
        try:
            if equation(0) * equation(remaining) < 0:
                x_opt = brentq(equation, 0, remaining)
            elif abs(equation(0)) < abs(equation(remaining)): x_opt = 0
            else: x_opt = remaining
        except:
            x_opt = 0
            
        return max(0, x_opt), "Stage 2"

# ==============================================================================
# HÀM CHẠY MÔ PHỎNG (CÓ HIỂN THỊ GIAI ĐOẠN)
# ==============================================================================
def run_simulation_vnd_staged():
    # --- CẤU HÌNH ---
    Q = 400
    m = 30.0
    M = 45.0
    h_cost = 0.05
    a_param = 250
    b_param = 5

    prices = [
        45.0, 44.0, 43.0,       # Tuần đầu
        41.0, 40.0, 39.5, 38.0, # Giảm nhiệt
        37.0, 36.0, 35.0, 34.0, # Ổn định
        32.0, 31.0, 30.0, 29.0  # Cuối vụ
    ]
    
    print("\n" + "="*95)
    print(f"MÔ PHỎNG BÁN IPHONE 16 (CÓ CHI PHÍ LƯU KHO & HIỂN THỊ GIAI ĐOẠN)")
    print(f"Vốn: {Q} máy | Giá: {m}-{M} tr | Phí: {h_cost*1000:,.0f}k/ngày")
    print("="*95)
    
    sim = HoldingCostSimulator(Q, m, M, h_cost, a_param, b_param)
    y = 0
    total_net_profit = 0
    
    # Header bảng (Thêm cột Giai đoạn)
    print(f"{'Ngày':<5} {'Giá TT':<10} {'Bán (Cây)':<12} {'Doanh Thu':<12} {'Phí Tồn':<12} {'Lãi Ròng':<12} {'Giai đoạn':<12}")
    print("-" * 95)
    
    for t, p in enumerate(prices):
        day = t + 1
        is_last = (t == len(prices) - 1)
        
        # Gọi hàm decide mới (trả về cả x và stage)
        x, stage_name = sim.decide_with_stage(p, y, day, is_last)
        
        # Tính toán
        revenue = x * p
        holding_fee = x * (day - 1) * h_cost 
        net_profit = revenue - holding_fee
        
        y += x
        total_net_profit += net_profit
        
        # In ra màn hình
        print(f"{day:<5} {p:<10.1f} {x:<12.1f} {revenue:<12.1f} {holding_fee:<12.1f} {net_profit:<12.1f} {stage_name:<12}")
        
    print("-" * 95)
    print(f"TỔNG KẾT:")
    print(f"1. Tổng bán:      {y:.0f} / {Q} máy")
    print(f"2. Tổng Lãi Ròng: {total_net_profit:,.1f} Triệu VNĐ")

if __name__ == "__main__":
    run_simulation_vnd_staged()