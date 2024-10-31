class Context:
    # 定义一个静态变量，用于存储策略
    register_map = {}

    def __init__(self):
        pass

    @classmethod
    def register_strategy(cls, strategy, strategy_name):
        cls.register_map[strategy_name] = strategy

    @classmethod
    def get_strategy(cls, strategy_name):
        return cls.register_map.get(strategy_name)
