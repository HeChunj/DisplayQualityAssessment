'''
1. 采集图像预处理功能， 可以支持将采集导入的图像根据用户输入的亮色度值进行图像预处理（对齐，切分，位置对应）。
2. 计算对应图片指标，通过不同测试图像对被测样品的清晰度、图像噪声、白平衡、灰阶表现、色彩准确性、色彩饱和度、图像对比度、区域控光效果等计算其指标。
3. 设计软件界面， 形成友好的操作界面，可以让用户快速找到并完成图像导入、图像预处理、图像质量分析等操作，并查看评分。
'''

# 定义一个抽象策略类，实现Strategy接口
from Strategy.Context import Context


class AbstractStrategy:
    def register(self, stragety_name):
        print(f"Register {stragety_name} strategy")
        Context.register_strategy(self, stragety_name)
