import tushare as ts
import pandas as pd
import time
import datetime
import numpy as np
pro = ts.pro_api()

def initial_get_trading_data(start_date,end_date):
    """
    首次获取交易数据
    """
    # 获取股票基本信息并过滤ST股票
    df_basic = pro.stock_basic(fields=['ts_code','name','industry'])
    df_basic = df_basic[~df_basic['name'].str.contains('ST')]

    data_1 = []
    data_2 = []
    data_3 = []
    # ['000001.SZ','000002.SZ']
    for code in df_basic['ts_code']:
        daily_1 = pro.daily_basic(
            ts_code=code,
            start_date=start_date,
            end_date=end_date,
            fields=['ts_code','trade_date','turnover_rate','turnover_rate_f','volume_ratio','float_share','total_mv','circ_mv']
        )
        daily_2 = pro.daily(ts_code=code,
            start_date=start_date,
            end_date=end_date,
            fields=['ts_code','trade_date','open','high','low','close','pre_close','change','pct_chg','vol','amount'])
        daily_3 = pro.stk_limit(ts_code=code,
            start_date=start_date,
            end_date=end_date,
            fields=['ts_code', 'trade_date', 'up_limit','down_limit'])

        data_1.append(daily_1)
        data_2.append(daily_2)
        data_3.append(daily_3)
        print(f'获取成功: {code}')
        time.sleep(0.02)  # 添加延迟

    # 合并数据
    df_daily1 = pd.concat(data_1)
    df_daily2 = pd.concat(data_2)
    df_daily3 = pd.concat(data_3)

    # 三个表合并
    df_daily = pd.merge(df_daily1, df_daily2, on=['ts_code','trade_date'])
    df_daily = pd.merge(df_daily, df_daily3, on=['ts_code','trade_date'])

    # 再合并基础信息(使用左连接保留所有基础信息)
    df_merged = pd.merge(df_basic, df_daily, on='ts_code')

    df_merged.to_csv("D:\\Coding_Programs\\Quant\\test.csv", index=False)
    print('数据保存成功')

    '''
    导出的数据结构是
    ts_code	name	industry	trade_date	close	turnover_rate	total_mv	circ_mv	amount	pct_change
    000001.SZ	平安银行	银行	20241231	11.7	0.7603	22704924.29	22704572.51	1747242.074	-0.020920502
    000001.SZ	平安银行	银行	20241230	11.95	0.6966	23190072.25	23189712.95	1610892.096	0.01014370245139462
    000001.SZ	平安银行	银行	20241227	11.83	0.6648	22957201.23	22956845.54	1518383.345	-0.002529511
    000001.SZ	平安银行	银行	20241226	11.86	0.5154	23015418.98	23015062.39	1183745.519	-0.005033557
    .
    .
    .
    '''
initial_get_trading_data(20220101,20250516)


def add_trading_data(end_date):
    """
    传入要补充交易数据的结束时间(str)
    新获取的数据会与原来的交易数据表格合并补充
    返回合并完成的表格数据
    """
    # 先导入原有的交易数据
    data = pd.read_csv("D:\\Coding_Programs\\Quant\\Trading_Data.csv",index_col=0)
    # 获取原有交易数据的所有交易时间
    trade_date = data['trade_date'].unique()
    if np.int64(end_date) not in trade_date:
        # 原始数据的第一个日期加1天是开始日期,要转化为str
        start_date = (pd.to_datetime(str(trade_date[0]),format='%Y%m%d') + datetime.timedelta(days=1)).strftime('%Y%m%d')
        # 获取股票基本信息并过滤ST股票
        df_basic = pro.stock_basic(fields=['ts_code', 'name', 'industry'])
        df_basic = df_basic[~df_basic['name'].str.contains('ST')]

        data_1 = []
        data_2 = []
        data_3 = []
        for code in df_basic['ts_code']:
            daily_1 = pro.daily_basic(
                ts_code=code,
                start_date=start_date,
                end_date=end_date,
                fields=['ts_code', 'trade_date', 'turnover_rate', 'total_mv']
            )
            daily_2 = pro.daily(ts_code=code,
                                start_date=start_date,
                                end_date=end_date,
                                fields=['ts_code', 'trade_date', 'close', 'open', 'high', 'low', 'pct_change',
                                        'amount'])
            daily_3 = pro.stk_limit(ts_code=code,
                                    start_date=start_date,
                                    end_date=end_date,
                                    fields=['ts_code', 'trade_date', 'up_limit', 'down_limit'])
            data_1.append(daily_1)
            data_2.append(daily_2)
            data_3.append(daily_3)
            print(f'获取成功: {code}')
            time.sleep(0.05)  # 添加延迟

        # 合并数据
        df_daily1 = pd.concat(data_1)
        df_daily2 = pd.concat(data_2)
        df_daily3 = pd.concat(data_3)

        # 先合并两个daily表
        df_daily = pd.merge(df_daily1, df_daily2, on=['ts_code', 'trade_date'])
        df_daily = pd.merge(df_daily, df_daily3, on=['ts_code', 'trade_date'])

        # 再合并基础信息(使用左连接保留所有基础信息)
        df_merged = pd.merge(df_basic, df_daily, on='ts_code')

        combined_df = pd.concat([data, df_merged], axis=0)

        # 合并排序之前先转成可比格式
        combined_df["trade_date"] = pd.to_numeric(combined_df["trade_date"])
        # 按股票代码分组后并行处理
        combined_df = combined_df.groupby("ts_code", sort=False).apply(
            lambda x: x.sort_values("trade_date", ascending=False)
        ).reset_index(drop=True)

        # 查看结果
        print(combined_df.head(10))
        # 保存结果
        combined_df.to_csv("D:\\Coding_Programs\\Quant\\Trading_Data.csv", index=False)
        print('数据保存成功')

        return combined_df

    else:
        print('此时间的数据已经存在')
        return None
