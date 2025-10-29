import numpy as np
from quantylab.rltrader import utils


class Agent:
    # 에이전트 상태가 구성하는 값 개수
    # 주식 보유 비율, 현재 손익, 평균 매수 단가 대비 등락률
    STATE_DIM = 3

    # 매매 수수료 및 세금
    TRADING_CHARGE = 0.00015  # 거래 수수료 0.015%
    # TRADING_CHARGE = 0.00011  # 거래 수수료 0.011%
    # TRADING_CHARGE = 0  # 거래 수수료 미적용
    TRADING_TAX = 0.0025  # 거래세 0.25%
    # TRADING_TAX = 0  # 거래세 미적용

    # 행동
    ACTION_BUY = 0  # 매수
    ACTION_SELL = 1  # 매도
    ACTION_HOLD = 2  # 관망
    # 인공 신경망에서 확률을 구할 행동들
    ACTIONS = [ACTION_BUY, ACTION_SELL, ACTION_HOLD]
    NUM_ACTIONS = len(ACTIONS)  # 인공 신경망에서 고려할 출력값의 개수

    def __init__(self, environment, initial_balance, min_trading_price, max_trading_price):
        # 현재 주식 가격을 가져오기 위해 환경 참조
        self.environment = environment
        self.initial_balance = initial_balance  # 초기 자본금

        # 최소 단일 매매 금액, 최대 단일 매매 금액
        self.min_trading_price = min_trading_price
        self.max_trading_price = max_trading_price

        # Agent 클래스의 속성
        self.balance = initial_balance  # 현재 현금 잔고
        self.num_stocks = 0  # 보유 주식 수
        # 포트폴리오 가치: balance + num_stocks * {현재 주식 가격}
        self.portfolio_value = 0
        self.num_buy = 0  # 매수 횟수
        self.num_sell = 0  # 매도 횟수
        self.num_hold = 0  # 관망 횟수

        # Agent 클래스의 상태
        self.ratio_hold = 0  # 주식 보유 비율
        self.profitloss = 0  # 현재 손익
        self.avg_buy_price = 0  # 주당 매수 단가
        # 최소 거래 단위(수량, 주식 수 기준). 암호화폐는 소수점 거래 가능하므로
        # 기본값을 작은 소수로 설정. 단위는 '주식 수'이며, 실제 거래 금액은
        # trading_unit * price로 계산됩니다.
        self.min_trading_unit = 0.000001

    def reset(self):
        self.balance = self.initial_balance
        self.num_stocks = 0
        self.portfolio_value = self.initial_balance
        self.num_buy = 0
        self.num_sell = 0
        self.num_hold = 0
        self.ratio_hold = 0
        self.profitloss = 0
        self.avg_buy_price = 0

    def set_balance(self, balance):
        self.initial_balance = balance

    def get_states(self):
        # protect against division by zero when portfolio_value is 0
        price = self.environment.get_price()
        if self.portfolio_value <= 0 or price <= 0:
            self.ratio_hold = 0
        else:
            self.ratio_hold = self.num_stocks * price / self.portfolio_value
        return (
            self.ratio_hold,
            self.profitloss,
            (self.environment.get_price() / self.avg_buy_price) - 1 \
                if self.avg_buy_price > 0 else 0
        )

    def decide_action(self, pred_value, pred_policy, epsilon):
        confidence = 0.

        pred = pred_policy
        if pred is None:
            pred = pred_value

        if pred is None:
            # 예측 값이 없을 경우 탐험
            epsilon = 1
        else:
            # 값이 모두 같은 경우 탐험
            maxpred = np.max(pred)
            if (pred == maxpred).all():
                epsilon = 1

            # if pred_policy is not None:
            #     if np.max(pred_policy) - np.min(pred_policy) < 0.05:
            #         epsilon = 1

        # 탐험 결정
        if np.random.rand() < epsilon:
            exploration = True
            action = np.random.randint(self.NUM_ACTIONS)
        else:
            exploration = False
            action = np.argmax(pred)

        confidence = .5
        if pred_policy is not None:
            confidence = pred[action]
        elif pred_value is not None:
            confidence = utils.sigmoid(pred[action])

        return action, confidence, exploration

    def validate_action(self, action):
        if action == Agent.ACTION_BUY:
            # 암호화폐의 경우 소수점 거래를 허용하므로 최소 단위로 살 수 있는지 확인
            if self.balance < self.environment.get_price() * (1 + self.TRADING_CHARGE) * self.min_trading_unit:
                return False
        elif action == Agent.ACTION_SELL:
            # 주식 잔고가 있는지 확인
            if self.num_stocks <= 0:
                return False
        return True

    def decide_trading_unit(self, confidence):
        # 반환값: 거래할 '수량' (부동소수)
        if np.isnan(confidence):
            # 최소 단위로 거래
            return self.min_trading_unit

        # confidence에 따라 화폐 단위로 거래 금액을 산정
        trade_amount_money = self.min_trading_price + confidence * (self.max_trading_price - self.min_trading_price)
        # 거래 수량 = 금액 / 현재 가격
        price = self.environment.get_price()
        # guard: avoid division by zero or invalid price
        if price is None or price <= 0:
            return self.min_trading_unit
        unit = trade_amount_money / price
        # 제한: 최소 단위 이상, 최대 단위 이하
        unit = max(unit, self.min_trading_unit)
        unit = min(unit, max(self.max_trading_price / price, unit))
        return unit

    def act(self, action, confidence):
        if not self.validate_action(action):
            action = Agent.ACTION_HOLD

        # 이전 포트폴리오 가치 저장 (보상 계산용)
        prev_portfolio_value = self.portfolio_value

        # 환경에서 현재 가격 얻기
        curr_price = self.environment.get_price()

        # 매수
        if action == Agent.ACTION_BUY:
            trading_unit = float(self.decide_trading_unit(confidence))
            # 사용할 수 있는 최대 단위 (잔고/가격 기준)
            max_affordable = self.balance / (curr_price * (1 + self.TRADING_CHARGE))
            max_by_price = self.max_trading_price / curr_price
            # 실제 거래 단위는 affordability와 가격 제한을 고려
            trading_unit = min(trading_unit, max_affordable, max_by_price)
            # 수수료를 적용하여 총 매수 금액 산정
            invest_amount = curr_price * (1 + self.TRADING_CHARGE) * trading_unit
            if invest_amount > 0 and trading_unit >= self.min_trading_unit:
                # avg_buy_price 업데이트 (부동소수 지원)
                if self.num_stocks + trading_unit > 0:
                    self.avg_buy_price = (
                        (self.avg_buy_price * self.num_stocks + curr_price * trading_unit)
                        / (self.num_stocks + trading_unit)
                    )
                else:
                    self.avg_buy_price = curr_price
                self.balance -= invest_amount
                self.num_stocks += trading_unit
                self.num_buy += 1

        # 매도
        elif action == Agent.ACTION_SELL:
            trading_unit = float(self.decide_trading_unit(confidence))
            # 보유 주식이 모자랄 경우 가능한 만큼 최대한 매도
            trading_unit = min(trading_unit, self.num_stocks)
            invest_amount = curr_price * (1 - (self.TRADING_TAX + self.TRADING_CHARGE)) * trading_unit
            if invest_amount > 0 and trading_unit >= self.min_trading_unit:
                # avg_buy_price 업데이트
                if self.num_stocks - trading_unit > 0:
                    self.avg_buy_price = (
                        (self.avg_buy_price * self.num_stocks - curr_price * trading_unit)
                        / (self.num_stocks - trading_unit)
                    )
                else:
                    self.avg_buy_price = 0
                self.num_stocks -= trading_unit
                self.balance += invest_amount
                self.num_sell += 1

        # 관망
        elif action == Agent.ACTION_HOLD:
            self.num_hold += 1  # 관망 횟수 증가

        # 포트폴리오 가치 갱신
        self.portfolio_value = self.balance + curr_price * self.num_stocks
        self.profitloss = self.portfolio_value / self.initial_balance - 1

        # 보상: 직전 단계 대비 포트폴리오 가치 변화량을 초기 자본으로 정규화
        reward = (self.portfolio_value - prev_portfolio_value) / float(self.initial_balance)
        return reward
