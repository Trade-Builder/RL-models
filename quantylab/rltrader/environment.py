class Environment:
    PRICE_IDX = 4  # 종가의 위치

    def __init__(self, chart_data=None):      
        self.chart_data = chart_data
        self.observation = None
        self.idx = -1
        
        # 초기 자본금 설정 (원화 기준)
        self.initial_balance = 10000000  # 1000만원
        
        # 수수료 설정 (업비트 기준 0.05%)
        self.TRADING_CHARGE = 0.0005
        
    def reset(self):
        self.observation = None
        self.idx = -1
        
    def observe(self):
        if len(self.chart_data) > self.idx + 1:
            self.idx += 1
            self.observation = self.chart_data.iloc[self.idx]
            return self.observation
        return None
    
    def get_price(self):
        if self.observation is not None:
            return self.observation['close']
        return 0
