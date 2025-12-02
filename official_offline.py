import numpy as np
from scipy.stats import truncnorm
from scipy.optimize import brentq

# --- CẤU HÌNH INPUT ---
Q = 200
prices = [38, 35, 32, 36, 37] # Đơn vị: Triệu VNĐ
a, b, delta = 100, 1, 0.2

# Cấu hình phân phối chuẩn cụt (Truncated Normal)
a_trunc, b_trunc = (1 - delta - 1) / 1.0, (1 + delta - 1) / 1.0
dist = truncnorm(a_trunc, b_trunc, loc=1, scale=1)

def calculate_x(p, lam):
    """Tính lượng bán x dựa trên giá p và giá sàn lambda"""
    if p <= lam: return 0.0
    # Fix lỗi toán học bằng cách kẹp giá trị xác suất trong (0, 1)
    val = max(0.0, min(0.9999999, 1 - lam/p))
    return (a - b*p) * dist.ppf(val)

def solve_offline_clean(Q, prices):
    # 1. Tìm Lambda (Giá sàn) sao cho Tổng bán = Q
    # Hàm mục tiêu: Q - Tổng(x) = 0
    objective = lambda lam: Q - sum(calculate_x(p, lam) for p in prices)
    
    try:
        # Tìm nghiệm trong khoảng [0, giá cao nhất]
        lam_opt = brentq(objective, 0, max(prices), xtol=1e-6)
    except:
        lam_opt = 0.0 # Fallback nếu lỗi

    # 2. Tính toán kết quả cuối cùng
    results = []
    total_sold = 0
    total_revenue = 0

    # Cập nhật Header bảng để hiển thị đơn vị
    print(f"{'Kỳ':<5} {'Giá (tr)':<10} {'Lượng bán(Sản phẩm)':<15} {'D.Thu  (tr)':<15}")
    print("-" * 50)

    for i, p in enumerate(prices):
        x = calculate_x(p, lam_opt)
        rev = p * x
        
        results.append((i+1, p, x, rev))
        total_sold += x
        total_revenue += rev
        
        # In từng dòng
        print(f"{i+1:<5} {p:<10} {x:<15.2f} {rev:<15.2f}")

    print("-" * 50)
    print(f"GIÁ SÀN (Shadow Price): {lam_opt:.4f}")
    print(f"TỔNG SỐ LƯỢNG BÁN:      {total_sold:.2f} / {Q} sản phẩm")
    # Thêm đơn vị vào tổng doanh thu
    print(f"TỔNG DOANH THU:         {total_revenue:,.2f} Triệu VNĐ")

# --- CHẠY ---
solve_offline_clean(Q, prices)