import numpy as np
from scipy.stats import truncnorm
from scipy.optimize import brentq

# ==============================================================================
# CLASS MÔ PHỎNG THUẬT TOÁN ALG-IR (Logic giữ nguyên)
# ==============================================================================
class OnlineBatchSimulator:
    def __init__(self, Q, m, M, a=100, b=1, delta=0.2):
        self.Q_init = Q
        self.m = m
        self.M = M
        self.a = a
        self.b = b
        
        # Tỷ lệ giá trần/sàn (Theta)
        self.theta = M / m
        
        # [QUAN TRỌNG] Tính Ngưỡng chuyển đổi (Threshold)
        self.threshold = self.Q_init / (1 + np.log(self.theta))
        
        # Cấu hình phân phối xác suất
        a_trunc, b_trunc = (1 - delta - 1) / 1.0, (1 + delta - 1) / 1.0
        self.dist = truncnorm(a_trunc, b_trunc, loc=1, scale=1)

    def _F_inverse(self, val):
        """Hàm nghịch đảo F^-1"""
        val = max(0.0000001, min(0.9999999, val)) 
        return self.dist.ppf(val)

    def _phi(self, z):
        """Hàm phạt chi phí biên giả định"""
        if z <= self.threshold:
            return self.m
        else:
            exponent = ((1 + np.log(self.theta)) / self.Q_init) * z - 1
            return self.m * np.exp(exponent)

    def _marginal_revenue_prime(self, x, p):
        """Đạo hàm doanh thu biên"""
        D_mean = self.a - self.b * p
        if x <= D_mean * 0.8: return p
        if x >= D_mean * 1.2: return 0
        return p * (1 - self.dist.cdf(x / D_mean))

    def decide_retrieval(self, p, current_y, is_last_period):
        """
        Luồng xử lý chính
        """
        # 1. Xử lý kỳ cuối cùng
        if is_last_period:
            return max(0, self.Q_init - current_y)

        # 2. Tính thử nghiệm Giai đoạn 1
        D_base = self.a - self.b * p
        val_prob = 1 - (self.m / p)
        
        if val_prob <= 0: x_trial = 0
        else: x_trial = D_base * self._F_inverse(val_prob)

        # 3. Kiểm tra điều kiện Ngưỡng
        if current_y + x_trial <= self.threshold:
            return max(0, x_trial)
        
        # 4. Chuyển sang Giai đoạn 2: Giải phương trình
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
# HÀM CHẠY (RUNNER - Đã chuẩn hóa hiển thị Tiền)
# ==============================================================================
def run_online_simulation(Q, price_list, m, M):
    simulator = OnlineBatchSimulator(Q, m, M)
    
    y_accumulated = 0
    total_revenue = 0
    
    # Hiển thị thông tin đơn vị
    print(f"\n>>> CHẠY MÔ PHỎNG (Đơn vị: Triệu VNĐ) VỚI Q={Q}, GIÁ=[{m}, {M}]")
    print(f"Ngưỡng chuyển đổi: {simulator.threshold:.2f}")
    
    # Header bảng (Cập nhật tiêu đề cột)
    print("-" * 85)
    print(f"{'Ngày':<5} {'Giá (Tr)':<10} {'Quyết định':<15} {'D.Thu (Tr)':<15} {'Kho còn':<15} {'Giai đoạn':<15}")
    print("-" * 85)
    
    for t, p in enumerate(price_list):
        is_last = (t == len(price_list) - 1)
        
        # Gọi thuật toán
        x = simulator.decide_retrieval(p, y_accumulated, is_last)
        
        # Cập nhật số liệu
        revenue = x * p
        y_accumulated += x
        total_revenue += revenue
        remaining = Q - y_accumulated
        
        # Xác định tên giai đoạn
        if is_last: gd = "Xả kho"
        elif y_accumulated <= simulator.threshold: gd = "Stage 1"
        else: gd = "Stage 2"
        
        # In ra màn hình
        print(f"{t+1:<5} {p:<10} {x:<15.2f} {revenue:<15.2f} {remaining:<15.2f} {gd:<15}")

    print("-" * 85)
    print(f"TỔNG DOANH THU: {total_revenue:,.2f} Triệu VNĐ")
    print(f"TỔNG ĐÃ BÁN:    {y_accumulated:.2f} / {Q}")

# ==============================================================================
# MAIN (CHẠY THỬ - Số liệu giữ nguyên)
# ==============================================================================
if __name__ == "__main__":
    # Kịch bản bài báo
    run_online_simulation(Q=200, price_list=[38, 35, 32, 36, 37], m=30, M=40)
    
    # Kịch bản ngẫu nhiên (Ví dụ iPhone)
    run_online_simulation(Q=200, price_list=[42, 41, 38, 35, 33, 34, 39, 36, 32, 30], m=30, M=45)