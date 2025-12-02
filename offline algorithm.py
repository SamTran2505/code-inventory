import numpy as np
from scipy.stats import truncnorm
from scipy.optimize import brentq

# ==============================================================================
# CẤU HÌNH INPUT (ĐÚNG YÊU CẦU: Q=500, a=80, b=1)
# ==============================================================================
Q = 500  
#  12 mức giá từ 30 đến 45 (Triệu VNĐ)
 # Giá 12 kì 30-45tr
prices = [
        45.0, 44.0, 43.0,  # Tuần đầu
        41.0, 40.0, 39.5,  # Giảm nhiệt
        37.0, 36.0, 35.0,  # Ổn định
        32.0, 31.0, 30.0,  # Cuối vụ
    ]
# Tham số cầu: a=80, b=1 (Để tổng cầu ~ 540, sát với Q=500)
a, b, delta = 80, 1, 0.2

# Cấu hình phân phối chuẩn cụt
a_trunc, b_trunc = (1 - delta - 1) / 1.0, (1 + delta - 1) / 1.0
dist = truncnorm(a_trunc, b_trunc, loc=1, scale=1)

def calculate_x(p, lam):
    """Tính lượng bán x dựa trên giá p và giá sàn lambda"""
    if p <= lam: return 0.0
    val = max(0.0, min(0.9999999, 1 - lam/p))
    return (a - b*p) * dist.ppf(val)

def solve_offline_clean(Q, prices):
    # 1. Tìm Lambda (Giá sàn) sao cho Tổng bán = Q
    objective = lambda lam: Q - sum(calculate_x(p, lam) for p in prices)
    
    try:
        lam_opt = brentq(objective, 0, max(prices), xtol=1e-6)
    except:
        lam_opt = 0.0 

    # 2. Tính toán và Hiển thị kết quả
    total_sold = 0
    total_revenue = 0

    print(f"\n--- KẾT QUẢ TỐI ƯU OFFLINE (Q={Q}) ---")
    # Header bảng được căn chỉnh đẹp
    print("-" * 60)
    print(f"{'Kỳ':<5} {'Giá (Tr)':<10} {'Bán (SP)':<15} {'D.Thu (Tr)':<15}")
    print("-" * 60)

    for i, p in enumerate(prices):
        x = calculate_x(p, lam_opt)
        rev = p * x
        
        total_sold += x
        total_revenue += rev
        
        # In từng dòng với format 2 chữ số thập phân
        print(f"{i+1:<5} {p:<10} {x:<15.2f} {rev:<15,.2f}")

    print("-" * 60)
    print(f"GIÁ SÀN (Shadow Price): {lam_opt:.2f} Triệu")
    print(f"TỔNG SỐ LƯỢNG BÁN:      {total_sold:.2f} / {Q} sản phẩm")
    print(f"TỔNG DOANH THU:         {total_revenue:,.2f} Triệu VNĐ")

# --- CHẠY ---
if __name__ == "__main__":
    solve_offline_clean(Q, prices)