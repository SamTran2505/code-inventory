import numpy as np
from scipy.stats import truncnorm
from scipy.optimize import brentq

# ==============================================================================
# CLASS MÔ PHỎNG THUẬT TOÁN ALG-IR (Logic giữ nguyên)
# ==============================================================================
class OnlineBatchSimulator:
    def __init__(self, Q, m, M, a, b, delta=0.2):
        self.Q_init = Q
        self.m = m
        self.M = M
        self.a = a
        self.b = b
        
        self.theta = M / m
        self.threshold = self.Q_init / (1 + np.log(self.theta))
        
        a_trunc, b_trunc = (1 - delta - 1) / 1.0, (1 + delta - 1) / 1.0
        self.dist = truncnorm(a_trunc, b_trunc, loc=1, scale=1)

    def _F_inverse(self, val):
        val = max(0.0000001, min(0.9999999, val)) 
        return self.dist.ppf(val)

    def _phi(self, z):
        if z <= self.threshold:
            return self.m
        else:
            exponent = ((1 + np.log(self.theta)) / self.Q_init) * z - 1
            return self.m * np.exp(exponent)

    def _marginal_revenue_prime(self, x, p):
        D_mean = self.a - self.b * p
        if x <= D_mean * 0.8: return p
        if x >= D_mean * 1.2: return 0
        return p * (1 - self.dist.cdf(x / D_mean))

    def decide_retrieval(self, p, current_y, is_last_period):
        if is_last_period:
            return max(0, self.Q_init - current_y)

        D_base = self.a - self.b * p
        val_prob = 1 - (self.m / p)
        
        if val_prob <= 0: x_trial = 0
        else: x_trial = D_base * self._F_inverse(val_prob)

        if current_y + x_trial <= self.threshold:
            return max(0, x_trial)
        
        def equation(x):
            return self._marginal_revenue_prime(x, p) - self._phi(current_y + x)

        remaining = self.Q_init - current_y
        try:
            if equation(0) * equation(remaining) < 0:
                x_opt = brentq(equation, 0, remaining)
            elif abs(equation(0)) < abs(equation(remaining)): x_opt = 0
            else: x_opt = remaining
        except:
            x_opt = 0
            
        return max(0, x_opt)

# ==============================================================================
# HÀM CHẠY
# ==============================================================================
def run_online_simulation(Q, price_list, m, M, a, b):
    simulator = OnlineBatchSimulator(Q, m, M, a, b)
    y_accumulated = 0
    total_revenue = 0
    
    print(f"\n>>> CHẠY MÔ PHỎNG ALG-IR (Q={Q})")
    print(f"    Ngưỡng chuyển đổi: {simulator.threshold:.2f}")
    
    print("-" * 85)
    print(f"{'Kỳ':<5} {'Giá (Tr)':<10} {'Bán (SP)':<15} {'D.Thu (Tr)':<15} {'Kho còn':<15} {'Giai đoạn':<15}")
    print("-" * 85)
    
    for t, p in enumerate(price_list):
        is_last = (t == len(price_list) - 1)
        x = simulator.decide_retrieval(p, y_accumulated, is_last)
        
        revenue = x * p
        y_accumulated += x
        total_revenue += revenue
        remaining = Q - y_accumulated
        
        if is_last: gd = "Xả kho"
        elif y_accumulated <= simulator.threshold: gd = "Stage 1"
        else: gd = "Stage 2"
        
        print(f"{t+1:<5} {p:<10} {x:<15.2f} {revenue:<15.2f} {remaining:<15.2f} {gd:<15}")

    print("-" * 85)
    print(f"TỔNG DOANH THU: {total_revenue:,.2f} Triệu VNĐ")
    print(f"TỔNG ĐÃ BÁN:    {y_accumulated:.2f} / {Q} sản phẩm")

# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == "__main__":
    Q_val = 500  
    m_val = 30.0 
    M_val = 45.0 
    a_val = 80
    b_val = 1
    
    # Giá 12 kì 30-45tr
prices = [
        45.0, 44.0, 43.0,  # Tuần đầu
        41.0, 40.0, 39.5,  # Giảm nhiệt
        37.0, 36.0, 35.0,  # Ổn định
        32.0, 31.0, 30.0,  # Cuối vụ
]
    
    
run_online_simulation(Q_val, prices, m_val, M_val, a_val, b_val)