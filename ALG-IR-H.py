import numpy as np
from scipy.stats import truncnorm
from scipy.optimize import brentq

# ==============================================================================
# CODE 2: ALGORITHM 2 (TỐI ƯU LỢI NHUẬN - CÓ PHÍ LƯU KHO)
# ==============================================================================
class ALG_IR_H_Simulator:
    def __init__(self, Q, m, M, h, a, b, delta=0.2):
        self.Q_init = Q
        self.m = m
        self.M = M
        self.h = h  # Chi phí lưu kho
        self.a = a
        self.b = b
        
        a_trunc, b_trunc = (1 - delta - 1) / 1.0, (1 + delta - 1) / 1.0
        self.dist = truncnorm(a_trunc, b_trunc, loc=1, scale=1)

    def _F_inverse(self, val):
        val = max(1e-6, min(1-1e-6, val))
        return self.dist.ppf(val)
    
    def _F_cdf(self, val):
        return self.dist.cdf(val)

    def _get_dynamic_params(self, t):
        """[ALG 2] Tham số thay đổi theo thời gian t"""
        holding_deduction = (t - 1) * self.h
        eff_m = self.m - holding_deduction
        eff_M = self.M - holding_deduction
        
        if eff_m <= 0.1: return 1000, 0.1 
        theta_t = eff_M / eff_m
        return theta_t, eff_m

    def _phi_h(self, z, t):
        theta_t, eff_m = self._get_dynamic_params(t)
        threshold_t = self.Q_init / (1 + np.log(theta_t))
        
        if z <= threshold_t: return eff_m
        else:
            exponent = ((1 + np.log(theta_t)) / self.Q_init) * z - 1
            return eff_m * np.exp(exponent)

    def _pi_prime_h(self, x, p, t):
        """Lợi nhuận biên ròng (Đã trừ phí lưu kho tích lũy)"""
        holding_deduction = (t - 1) * self.h
        D_mean = self.a - self.b * p
        
        if x <= D_mean * 0.8: mr = p
        elif x >= D_mean * 1.2: mr = 0
        else: mr = p * (1 - self._F_cdf(x / D_mean))
        
        return mr - holding_deduction

    def decide(self, p, current_y, t, is_last):
        if is_last:
            return max(0, self.Q_init - current_y), "Xả kho"

        theta_t, _ = self._get_dynamic_params(t)
        threshold_t = self.Q_init / (1 + np.log(theta_t))

        D_mean = self.a - self.b * p
        val_prob = 1 - (self.m / p) 
        if val_prob <= 0: x_trial = 0
        else: x_trial = D_mean * self._F_inverse(val_prob)

        if current_y + x_trial <= threshold_t:
            return max(0, x_trial), "Stage 1"
        
        def equation(x):
            return self._pi_prime_h(x, p, t) - self._phi_h(current_y + x, t)

        remaining = self.Q_init - current_y
        try:
            if equation(0) * equation(remaining) < 0:
                x_opt = brentq(equation, 0, remaining)
            elif abs(equation(0)) < abs(equation(remaining)): x_opt = 0
            else: x_opt = remaining
        except: x_opt = 0
            
        return max(0, x_opt), "Stage 2"

def run_alg2():
    # --- CẤU HÌNH ---
    Q = 500
    m, M = 30.0, 40.0
    a, b = 80, 1
    h = 0.05 # Phí lưu kho
    prices = [38, 35, 33, 36, 32, 33, 38, 34, 31, 32, 31, 30]

    print("\n" + "="*80)
    print(f"ALGORITHM 2: TỐI ƯU LỢI NHUẬN (CÓ PHÍ LƯU KHO h={h})")
    print("="*80)
    
    sim = ALG_IR_H_Simulator(Q, m, M, h, a, b)
    y = 0
    total_revenue = 0
    total_net_profit = 0
    
    print(f"{'Kỳ':<5} {'Giá':<10} {'Bán (SP)':<12} {'D.Thu':<15} {'Lãi Ròng':<15} {'Giai đoạn':<12}")
    print("-" * 80)
    
    for t, p in enumerate(prices):
        day = t + 1
        is_last = (t == len(prices) - 1)
        x, stage = sim.decide(p, y, day, is_last)
        
        revenue = x * p
        holding_fee = x * (day - 1) * h
        net = revenue - holding_fee
        
        y += x
        total_revenue += revenue
        total_net_profit += net
        
        print(f"{day:<5} {p:<10.1f} {x:<12.2f} {revenue:<15,.2f} {net:<15,.2f} {stage:<12}")
        
    print("-" * 80)
    print(f"TỔNG KẾT ALG-2:")
    print(f"Tổng bán:      {y:.2f} / {Q}")
    print(f"Tổng Lãi Ròng: {total_net_profit:,.2f} Triệu VNĐ")

if __name__ == "__main__":
    run_alg2()