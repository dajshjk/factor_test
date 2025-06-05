import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from data_process import *

plt.rcParams['font.sans-serif'] = ['SimSun']  # 设置宋体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def IC_test(processed_df,factor,next_period_return):
    # 计算每日IC（皮尔逊相关系数）和Rank IC（斯皮尔曼相关系数）
    # 提取相关系数矩阵第1行第2列的数据
    ic_series = processed_df.groupby("trade_date").apply(
        lambda x: x[[factor, next_period_return]].corr(method="pearson").iloc[0, 1]
    )
    rank_ic_series = processed_df.groupby("trade_date").apply(
        lambda x: x[[factor, next_period_return]].corr(method="spearman").iloc[0, 1]
    )

    '''
                        factor_value  next_period_return
    factor_value          -0.531587         0.018376
    next_period_return    -0.605196        -0.015663
    '''

    # IC统计指标
    ic_mean = ic_series.mean()
    ic_abs_mean = ic_series.abs().mean()
    ic_std = ic_series.std()
    ic_ir = ic_mean / ic_std
    
    rank_ic_mean = rank_ic_series.mean()
    rank_ic_abs_mean = rank_ic_series.abs().mean()
    rank_ic_std = rank_ic_series.std()
    rank_ic_ir = rank_ic_mean / rank_ic_std
    
    ic_positive_ratio = (ic_series > 0).mean()
    rank_ic_positive_ratio = (rank_ic_series > 0).mean()

    print(f"""
    IC统计指标 (皮尔逊):
    --------------------------------
    IC均值: {ic_mean:.4f}
    IC绝对值均值: {ic_abs_mean:.4f}
    IC标准差: {ic_std:.4f}
    ICIR: {ic_ir:.4f}
    IC>0比例: {ic_positive_ratio:.2%}

    Rank IC统计指标 (斯皮尔曼):
    --------------------------------
    Rank IC均值: {rank_ic_mean:.4f}
    Rank IC绝对值均值: {rank_ic_abs_mean:.4f}  
    Rank IC标准差: {rank_ic_std:.4f}
    Rank ICIR: {rank_ic_ir:.4f}
    Rank IC>0比例: {rank_ic_positive_ratio:.2%}
    """)

    # 可视化IC分布
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.histplot(ic_series, kde=True)
    plt.title("IC分布 (皮尔逊)")
    
    plt.subplot(1, 2, 2)
    sns.histplot(rank_ic_series, kde=True)
    plt.title("Rank IC分布 (斯皮尔曼)")
    
    plt.tight_layout()
    plt.show()

def factor_return_ttest(processed_df,factor,next_period_return):
    """对因子收益率序列进行T检验"""
    # 1. 计算每期截面回归的因子收益率和t值
    factor_t_values = []
    factor_returns = []
    
    # 首先确保trade_date是日期类型，按照日期遍历，获取截面数据date_df
    for date,date_df in processed_df.groupby('trade_date'):
        X = sm.add_constant(date_df[factor])  # 添加常数项
        y = date_df[next_period_return]
        model = sm.OLS(y, X).fit()   # 第一个参数endog被解释变量，第二个参数exog解释变量

        factor_t_values.append(model.tvalues.iloc[1])  # 一个截面的因子系数的t值
        factor_returns.append(model.params.iloc[1])  # 一个截面的因子收益率，即回归方程的回归系数β值
    
    factor_t_values = np.array(factor_t_values)
    factor_returns = np.array(factor_returns)
    
    # 2. 计算t值相关指标
    abs_t_mean = np.abs(factor_t_values).mean()
    t_gt_2_ratio = (np.abs(factor_t_values) > 2).mean()
    
    # 3. 对因子收益率序列进行t检验.检验因子收益是否显著不为0
    t_stat, p_value = stats.ttest_1samp(factor_returns, 0)
    
    print(f"""
    因子收益率T检验结果:
    --------------------------------
    截面回归期数: {len(factor_t_values)}
    |t|均值: {abs_t_mean:.4f}
    |t|>2比例: {t_gt_2_ratio:.2%}
    
    因子收益率序列检验:
    --------------------------------
    平均因子收益率: {factor_returns.mean():.4f}
    T统计量: {t_stat:.4f}
    P值: {p_value:.4f}
    """)
    
    # 可视化
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    sns.histplot(np.abs(factor_t_values), kde=True)
    plt.axvline(x=2, color='r', linestyle='--')
    plt.title("|t|值分布")
    plt.xlabel("|t|")
    
    plt.subplot(1, 2, 2)
    sns.histplot(factor_returns, kde=True)
    plt.axvline(x=0, color='r', linestyle='--')
    plt.title("因子收益率分布")
    plt.xlabel("收益率")
    
    plt.tight_layout()
    plt.show()

def select_rebalance_data(df, n):
    """
    筛选数据实现 n 天调仓
    传入处理过的数据表格和n天调仓周期
    返回只包含调仓周期的数据集
    """
    # 转datetime升序排列

    df = df.sort_values('trade_date', ascending=True)
    # 按步长n筛选
    trade_dates = list(df['trade_date'].unique())
    rebalance_dates = trade_dates[::n]
    rebalance_df = processed_df[processed_df['trade_date'].isin(rebalance_dates)].sort_values(by=['trade_date','ts_code'],ascending=True)

    return rebalance_df

def group_backtest(processed_df,factor,next_period_return,groups=5):
    '''
    每一天核算一次因子值,按照因子值大小排序分5组
    n天调仓
    并且输出回测年化收益率、夏普比率、信息比率、最大回撤、胜率
    '''

    # n天调仓
    n = int(next_period_return[-1])
    # 保留调仓日的数据
    rebalance_df = select_rebalance_data(processed_df,n)
    # 初始化分组收益率存储
    group_returns = {i: [] for i in range(1, groups+1)}
    
    # 按交易日分组处理
    for date, date_df in rebalance_df.groupby('trade_date'):
        # 按因子值升序排序并分5组（因子值最小值为第1组）
        date_df = date_df.sort_values(factor, ascending=True)
        # 分5组从0-4，＋1变成1-5组
        date_df['group'] = pd.qcut(date_df[factor], groups, labels=False) + 1
        
        # 计算每组下期收益率(按收盘价等权重计算)
        for group, group_df in date_df.groupby('group'):
            group_return = group_df[next_period_return].mean()
            group_returns[group].append(group_return)
    
    # 计算绩效指标
    results = {}
    print(group_returns)

    for group in group_returns:
        returns = pd.Series(group_returns[group])
        holding_days = len(returns)*n   # 调仓次数×持仓时间=总时间
        # 年化收益率 假设252个交易日
        annual_return = (1 + returns).prod()**(252 / holding_days) - 1

        # 夏普比率 无风险收益率假设为30年期国债收益率1.9% 波动率做年化
        sharpe = annual_return / returns.std() * np.sqrt(252)

        # 最大回撤
        cum_returns = (1 + returns).cumprod()
        peak = cum_returns.expanding().max()
        drawdown = (cum_returns - peak) / peak
        max_drawdown = drawdown.min()
        
        # 胜率
        win_rate = (returns > 0).mean()
        
        results[group] = {
            '年化收益率': annual_return,
            '夏普比率': sharpe,
            '最大回撤': max_drawdown,
            '胜率': win_rate
        }
    
    # 输出结果
    print("\n分层回测结果:")
    print("--------------------------------")
    for group in sorted(results.keys()):
        print(f"第{group}组:")
        for metric in results[group]:
            if metric == '年化收益率':
                print(f"  {metric}: {results[group][metric]:.2%}")
            elif metric == '最大回撤':
                print(f"  {metric}: {results[group][metric]:.2%}")
            elif metric == '胜率':
                print(f"  {metric}: {results[group][metric]:.2%}")
            else:
                print(f"  {metric}: {results[group][metric]:.2f}")
        print("--------------------------------")
    
    # 绘制各组累积收益率曲线
    plt.figure(figsize=(12, 6))
    # 获取交易日期并转换为datetime格式
    trade_dates = pd.to_datetime(rebalance_df['trade_date'].unique(), format='%Y%m%d')
    trade_dates = np.sort(trade_dates)
    
    for group in sorted(group_returns.keys()):
        # 计算累积收益率
        cum_returns = (1 + pd.Series(group_returns[group])).cumprod() - 1
        # 确保日期和收益率数据长度一致
        if len(trade_dates) > len(cum_returns):
            trade_dates = trade_dates[:len(cum_returns)]
        # 绘制曲线
        plt.plot(trade_dates, cum_returns, label=f'第{group}组')
    
    plt.title('各组累积收益率对比')
    plt.xlabel('交易日期')
    plt.ylabel('累积收益率')
    plt.legend()
    plt.grid(True)
    # 优化日期显示
    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m'))
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    plt.show()
    
def process_single_factor_test(processed_df,factor_list,next_period_return):
    """
    传入处理好的数据，因子列表，下一期收益率的列名
    """
    processed_df['trade_date'] = pd.to_datetime(processed_df['trade_date'].astype(str), format='%Y%m%d')
    for factor in factor_list:
        IC_test(processed_df,factor,next_period_return)
        factor_return_ttest(processed_df,factor,next_period_return)
        group_backtest(processed_df,factor,next_period_return)


if __name__ == "__main__":
    processed_df = pd.read_csv('D:\\Coding_Programs\\Quant\\factor_test\\processed_data.csv')
    factor_list = ['turnover_rate']
    process_single_factor_test(processed_df, factor_list, 'next_period_return_5')

