import numpy as np
from scipy.stats import truncnorm
from scipy.optimize import brentq
import sys

# ==============================================================================
# CLASS TÍNH TOÁN (CORE LOGIC)
# ==============================================================================
class RealTimeAdvisor:
    def __init__(self, Q_total_initial, m, M, a=250, b=5, delta=0.2):
        """
        Lưu ý tham số mặc định cho iPhone (Đơn vị Triệu VNĐ):
        a=250, b=5 nghĩa là: 
          - Giá 30tr -> Nhu cầu = 250 - 5*30 = 100 khách
          - Giá 45tr -> Nhu cầu = 250 - 5*45 = 25 khách
        """
        self.Q_init = Q_total_initial
        self.m = m
        self.M = M
        self.a = a
        self.b = b
        self.theta = M / m
        
        # Tính Ngưỡng chuyển đổi (Threshold)
        self.threshold = self.Q_init / (1 + np.log(self.theta))
        
        # Cấu hình phân phối
        a_trunc, b_trunc = (1 - delta - 1) / 1.0, (1 + delta - 1) / 1.0
        self.dist = truncnorm(a_trunc, b_trunc, loc=1, scale=1)

    def _F_inverse(self, val):
        val = max(1e-6, min(1-1e-6, val))
        return self.dist.ppf(val)

    def _F_cdf(self, val):
        return self.dist.cdf(val)

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
        return p * (1 - self._F_cdf(x / D_mean))

    def get_advice(self, current_price, current_inventory, is_last_day=False):
        # 1. Kiểm tra tình huống đặc biệt
        if current_inventory <= 0:
            return 0.0, "HẾT HÀNG"
        if is_last_day:
            return current_inventory, "NGÀY CUỐI (Xả kho)"

        # 2. Tính toán vị thế
        y_accumulated = self.Q_init - current_inventory
        
        # 3. Tính toán Giai đoạn 1 (Thử nghiệm)
        D_base = self.a - self.b * current_price
        val_prob = 1 - (self.m / current_price)
        
        if val_prob <= 0:
            x_trial = 0
        else:
            x_trial = D_base * self._F_inverse(val_prob)

        # Nếu bán x_trial mà vẫn dưới ngưỡng -> Duyệt
        if y_accumulated + x_trial <= self.threshold:
            return max(0, x_trial), "Giai đoạn 1 (Bán theo sức mua)"
        
        # 4. Giai đoạn 2 (Hàng hiếm)
        def equation(x):
            return self._marginal_revenue_prime(x, current_price) - self._phi(y_accumulated + x)

        try:
            if equation(0) * equation(current_inventory) < 0:
                x_opt = brentq(equation, 0, current_inventory)
            elif abs(equation(0)) < abs(equation(current_inventory)): x_opt = 0
            else: x_opt = current_inventory
        except:
            x_opt = 0
            
        return max(0, x_opt), "Giai đoạn 2 (Hàng hiếm - Giữ giá)"

# ============================================================
# GIAO DIỆN TƯƠNG TÁC (Auto-Tracking)
# ============================================================

def run_smart_tool_iphone():
    print("==================================================")
    print("   TRỢ LÝ BÁN IPHONE 16 THÔNG MINH (ALG-IR)")
    print("   Đơn vị tính: TRIỆU VNĐ (VD: 30.5 = 30tr500)")
    print("==================================================")
    
    # 1. CẤU HÌNH BAN ĐẦU
    try:
        print("\n--- CẤU HÌNH ĐẦU MÙA VỤ ---")
        Q_init = float(input("1. Tổng nhập hàng ban đầu (VD: 200 máy): "))
        m = float(input("2. Giá sàn/Giá gốc (VD: 30.0 tr): "))
        M = float(input("3. Giá trần kỳ vọng (VD: 45.0 tr): "))
        
        # Tự động set tham số a, b cho iPhone nếu người dùng không rành
        advisor = RealTimeAdvisor(Q_init, m, M, a=250, b=5)
        print("-> Đã thiết lập mô hình cầu chuẩn cho iPhone (a=250, b=5)")
        
    except ValueError:
        print("Lỗi nhập liệu. Vui lòng nhập số.")
        return

    # BIẾN TRẠNG THÁI
    current_inventory = Q_init
    day_count = 1

    print(f"\n>>> BẮT ĐẦU CHIẾN DỊCH VỚI {current_inventory:.0f} MÁY <<<")
    
    while current_inventory > 0:
        print("\n" + "="*50)
        print(f"NGÀY THỨ {day_count} | Kho còn: {current_inventory:.1f} máy")
        print("="*50)
        
        try:
            # NHẬP DỮ LIỆU
            p_now = float(input(">> Giá thị trường hôm nay (triệu VNĐ)?: "))
            
            # Kiểm tra logic giá
            if p_now < 5: 
                print("(!) Cảnh báo: Bạn có nhập nhầm không? Giá iPhone thường > 20tr.")
                confirm = input("Ấn Enter để tiếp tục, 'n' để nhập lại: ")
                if confirm.lower() == 'n': continue

            last_input = input(">> Có phải ngày cuối cùng không? (y/n): ").strip().lower()
            is_last = (last_input == 'y')
            
            # TÍNH TOÁN
            amount_to_sell, reason = advisor.get_advice(p_now, current_inventory, is_last)
            
            # HIỂN THỊ KẾT QUẢ
            print("-" * 50)
            print(f"KẾT QUẢ: >>> NÊN BÁN {amount_to_sell:.1f} MÁY <<<")
            print(f"Lý do:   {reason}")
            print(f"Dự thu:  {amount_to_sell * p_now:,.2f} Triệu VNĐ")
            print("-" * 50)
            
            # TỰ ĐỘNG TRỪ KHO
            current_inventory -= amount_to_sell
            current_inventory = max(0, current_inventory)
            
            if is_last:
                print("\n=== KẾT THÚC MÙA VỤ (Đã xả kho) ===")
                break
                
            day_count += 1
            
        except ValueError:
            print("Lỗi: Vui lòng nhập số hợp lệ.")

    if current_inventory <= 0 and not is_last:
        print("\n=== CHÚC MỪNG! ĐÃ BÁN HẾT SẠCH HÀNG ===")

if __name__ == "__main__":
    run_smart_tool_iphone()