# factor_test 股票单因子检验

文件运行顺序：
1.data_collect.py
2.data_process.py
3.single_factor_test.py

1.从tushare获取原始数据，可以经过一系列的计算求得你想要的因子值
2.将计算过因子值和预测值的数据进行因子处理：去极值、标准化、中性化
3.将处理过因子值的数据进行单因子检验
