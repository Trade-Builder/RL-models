import random

def generate_stock_data(length=400, min_val=60000, max_val=150000):
    data = [random.uniform(min_val, max_val)]
    for _ in range(1, length):
        change = random.uniform(-8000, 8000)  # -1 ~ +1 변화
        new_value = data[-1] + change
        new_value = max(min_val, min(max_val, new_value))
        data.append(round(new_value, 2))
    return data

print(generate_stock_data())
