import torch
from typing import Union
import time


class Timeline:
    def __init__(self, coinname: str, starttime: Union[int, float], endtime: Union[int, float], file_path: str) -> None:
        """
        初始化 Timeline 类

        参数:
          coinname (str): 币种标识，例如 "AAVE"
          starttime (int or float): 开始时间（Unix绝对时间，单位毫秒）
          endtime (int or float): 结束时间（Unix绝对时间，单位毫秒）
          file_path (str): 包含K线数据的CSV文件路径

        CSV 文件格式示例:
          1704096000000,109.60000000,109.64000000,109.42000000,109.49000000,184.95000000,1704096299999,20260.22708000,128,50.05000000,5482.06786000,0

        本方法会读取在 starttime 与 endtime 范围内的记录，然后将这些记录转换为一个二维 torch 张量，
        其中每一行对应一根K线数据（即一个 12 维的向量），返回的 tensor 形状为 [k线数量, 12]。
        """
        self.coinname: str = coinname
        self.starttime: Union[int, float] = starttime
        self.endtime: Union[int, float] = endtime
        self.file_path: str = file_path
        self.tensor: torch.Tensor = self._load_data()
        # 新增：持仓量（[持仓量, 均价]），初始均设为0；交易记录为空二维向量，每条记录为 [时间戳, 价格, 交易量]；已实现利润设为0
        self.position: torch.Tensor = torch.tensor([0.0, 0.0], dtype=torch.float32)
        self.trading_record: torch.Tensor = torch.empty((0, 3), dtype=torch.float32)
        self.realized_profit: torch.Tensor = torch.tensor(0.0, dtype=torch.float32)

    def _load_data(self) -> torch.Tensor:
        data: list = []
        with open(self.file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",")
                try:
                    # 使用第一列作为时间戳进行过滤
                    open_time: float = float(parts[0])
                except ValueError:
                    continue
                if open_time < self.starttime or open_time > self.endtime:
                    continue
                try:
                    row = [float(x) for x in parts]
                except ValueError:
                    continue
                data.append(row)
        if len(data) == 0:
            # 无可用数据时返回空张量
            return torch.empty(0)
        # 转换为二维tensor，默认每一行为一根K线数据
        tensor: torch.Tensor = torch.tensor(data, dtype=torch.float32)
        return tensor

    def time_pass(self, new_rows: torch.Tensor) -> torch.Tensor:
        """
        向已有的 tensor 中添加新的行（新的K线数据）

        参数:
          new_rows (torch.Tensor): 形状必须是 [n, x]，其中 n 表示要添加的新K线数量，每一行是一根新的K线数据

        返回:
          更新后的 tensor
        """
        if self.tensor.numel() == 0:
            self.tensor = new_rows
        else:
            if new_rows.size(1) != self.tensor.size(1):
                raise ValueError(f"新增数据的列数 {new_rows.size(1)} 必须等于现有 tensor 的列数 {self.tensor.size(1)}")
            self.tensor = torch.cat((self.tensor, new_rows), dim=0)
        return self.tensor

    def increase_position(self, quantity: float) -> None:
        """
        加仓函数：根据当前最后一根 K 线的收盘价作为交易价格，增加仓位，并重新计算均价。
        
        参数:
          quantity (float): 加仓交易量（正数）
        
        更新:
          - 更新持仓: new_qty = old_qty + sign * quantity, 新均价 = (old_qty*old_avg + sign*quantity*trade_price) / new_qty
          - 记录交易：在交易记录中追加一条记录 [当前时间戳, trade_price, sign * quantity]
        """
        # 获取当前最后一根K线的收盘价，第5个值（索引4）
        trade_price: float = self.tensor[-1, 4].item()
        old_qty: float = self.position[0].item()
        old_avg: float = self.position[1].item()
        # 如果当前仓位为0，则默认建立多仓
        sign: float = 1.0 if old_qty >= 0 else -1.0
        new_qty: float = old_qty + sign * quantity
        if new_qty == 0:
            new_avg: float = 0.0
        else:
            new_avg = (old_qty * old_avg + sign * quantity * trade_price) / new_qty

        self.position = torch.tensor([new_qty, new_avg], dtype=torch.float32)

        # 添加交易记录
        record = torch.tensor([[time.time() * 1000, trade_price, sign * quantity]], dtype=torch.float32)
        if self.trading_record.numel() == 0:
            self.trading_record = record
        else:
            self.trading_record = torch.cat((self.trading_record, record), dim=0)

    def decrease_position(self, quantity: float) -> None:
        """
        减仓函数：根据当前最后一根 K 线的收盘价作为交易价格，减少仓位，不重新计算均价，
        且更新已实现利润，公式为：realized_profit += (trade_price - old_avg) * quantity * sign，
        其中 sign 为减仓前仓位的符号。
        
        参数:
          quantity (float): 减仓交易量（正数），不得超过当前仓位的绝对值
          
        更新:
          - 更新持仓：new_qty = old_qty - sign * quantity，如仓位变为0，均价置为0；
          - 更新已实现利润；
          - 记录交易：在交易记录中追加一条记录 [当前时间戳, trade_price, -sign * quantity]
        """
        trade_price: float = self.tensor[-1, 4].item()
        old_qty: float = self.position[0].item()
        old_avg: float = self.position[1].item()
        sign: float = 1.0 if old_qty >= 0 else -1.0
        if quantity > abs(old_qty):
            raise ValueError(f"减仓数量 {quantity} 超过当前持仓绝对值 {abs(old_qty)}")
        new_qty: float = old_qty - sign * quantity
        profit_change: float = (trade_price - old_avg) * quantity * sign
        self.realized_profit = torch.tensor(self.realized_profit.item() + profit_change, dtype=torch.float32)

        new_avg: float = old_avg if new_qty != 0 else 0.0
        self.position = torch.tensor([new_qty, new_avg], dtype=torch.float32)

        record = torch.tensor([[time.time() * 1000, trade_price, -sign * quantity]], dtype=torch.float32)
        if self.trading_record.numel() == 0:
            self.trading_record = record
        else:
            self.trading_record = torch.cat((self.trading_record, record), dim=0)

    def trade(self, quantity: float) -> None:
        """
        根据传入的交易量判断调用加仓或减仓函数：
          - 当持仓不为0时，如果交易量与持仓同符号，则调用加仓函数；否则调用减仓函数。
          - 当持仓为0时，根据交易量的符号建立对应的新仓位（正数建立多仓，负数建立空仓）。
          参数:
            quantity (float): 交易量。正数表示买入，加仓；负数表示卖出，减仓。
        """
        if quantity == 0:
            return
        current_qty: float = self.position[0].item()
        if current_qty == 0:
            # 当持仓为0时，根据交易量的符号直接建立新仓位
            trade_price: float = self.tensor[-1, 4].item()
            if quantity > 0:
                # 建立多仓
                self.increase_position(quantity)
            else:
                # 建立空仓（短仓）：直接设置仓位为负数量
                self.position = torch.tensor([quantity, trade_price], dtype=torch.float32)
                record = torch.tensor([[time.time() * 1000, trade_price, quantity]], dtype=torch.float32)
                if self.trading_record.numel() == 0:
                    self.trading_record = record
                else:
                    self.trading_record = torch.cat((self.trading_record, record), dim=0)
            return
        if current_qty * quantity > 0:
            # 同方向，加仓
            self.increase_position(abs(quantity))
        else:
            # 反方向，减仓
            self.decrease_position(abs(quantity))

if __name__ == "__main__":
    # 测试 Timeline 类，用于读取并处理 AAVE 的5分钟K线数据
    # 示例时间区间（请根据实际数据调整）
    start_time: int = 1704096000000
    end_time: int = start_time + 30000000  # 示例时间区间
    
    file_path: str = "/data/jucui/StatAgent/bn_data/data_bn/AAVE_klines_5m.csv"
    
    timeline: Timeline = Timeline("AAVE", start_time, end_time, file_path)
    print("Tensor shape:", timeline.tensor.shape)
    
    new: Timeline = Timeline("AAVE", start_time+30000000, end_time+30000000, file_path)
    timeline.time_pass(new.tensor)
    print("Tensor shape:", timeline.tensor.shape)
    
    # 新增示例：做多, 加仓两次，再全部减仓示例（每次操作之间模拟时间推进）
    def simulate_time_pass(new_trade_price: float):
        # 复制最后一根K线数据，并更新时间戳和收盘价
        last_row = timeline.tensor[-1].clone()
        last_row[0] = time.time() * 1000  # 更新时间戳
        last_row[4] = new_trade_price     # 设置新的收盘价
        new_row = last_row.unsqueeze(0)
        timeline.time_pass(new_row)

    print("做多, 加仓两次，再全部减仓示例:")
    # 模拟时间推进，设置交易价为 100
    simulate_time_pass(100)
    # 第一次加仓，买入 10
    timeline.trade(-10)
    print(timeline.realized_profit)
    print(timeline.position)

    # 模拟时间推进，设置交易价为 105
    simulate_time_pass(105)
    # 第二次加仓，买入 5
    timeline.trade(5)
    print(timeline.realized_profit)
    print(timeline.position)

    # 模拟时间推进，设置交易价为 110
    simulate_time_pass(110)
    # 全部减仓，根据当前持仓全部平仓
    current_position = timeline.position[0].item()
    timeline.trade(-current_position)

    print("交易记录:")
    print(timeline.trading_record)
    print("已实现利润:")
    print(timeline.realized_profit)
    