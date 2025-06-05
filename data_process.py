import pandas as pd
import statsmodels.api as sm

def get_trading_data(read_path,output_path):
    """
    传入交易数据地址和导出数据地址
    """
    df = pd.read_csv(read_path)
    output_path = output_path
    print(f"正在保存因子数据")

    return df,output_path

def get_industry_matrix(df):
    """
    传入原始交易数据，获取行业矩阵
    """
    # 先删除重复值，防止内存爆满，再把股票代码设置为index
    df = df[['ts_code', 'industry']].drop_duplicates().set_index('ts_code')
    # 指定某一列成为行，删除重复行
    df = pd.get_dummies(df.industry).drop_duplicates().astype(int)
    # 为index赋名ts_code
    df.index.name = 'ts_code'

    df.to_csv(output_path+'industry_matrix.csv')
    print('已完成industry_matrix写入')
    return df

def process_return(df,n=1):
    """
    传入原始交易数据，期望预测天数
    """
    df[f'next_period_return_{n}'] = df.groupby('ts_code')['close'].shift(n)/df['close'] - 1
    df = trading_data.dropna(subset=[f'next_period_return_{n}'])
    print(f'已处理下一期收益率--{n}天')

    return df,f'next_period_return_{n}'

def prepared_process(factor_list,df,df_industry,next_period_return):
    """
    合并保留股票代码、交易时间、多个因子值、总市值和行业矩阵
    ts_code,trade_date,turnover_rate,total_mv,IT设备,专用机械,中成药,乳制品,互联网..........
    """
    df = df[['ts_code','trade_date']+factor_list+['total_mv',next_period_return]]
    df = pd.merge(df, df_industry, on='ts_code')

    return df

def winsorize_mad(group, factor_name, n=5):
    """
    中位数去极值
    """
    median = group[factor_name].median()
    mad = (group[factor_name] - median).abs().median()
    high = median + n * mad
    low = median - n * mad
    group[factor_name] = group[factor_name].clip(lower=low, upper=high)
    return group

def standardize(group, factor_name):
    """
    z-score标准化
    """
    mean = group[factor_name].mean()
    std = group[factor_name].std()
    if std != 0:
        group[factor_name] = (group[factor_name] - mean) / std
    else:
        group[factor_name] = 0  # 避免除以0
    return group

def neutralize(group, factor_name,industry_columns):
    """
    使用线性回归进行行业中性化和市值中性化
    """

    X = sm.add_constant(group[industry_columns])
    y = group[factor_name]
    model = sm.OLS(y, X, hasconst=True).fit()
    group[factor_name] = model.resid

    X = sm.add_constant(group['total_mv'])
    y = group[factor_name]
    model = sm.OLS(y, X, hasconst=True).fit()
    group[factor_name] = model.resid

    return group

def process_factor_function(prepared_df,factor_list,industry_columns):
    """
    对每一行数据进行处理
    传入prepared_process合并后的数据,要处理的因子名称
    去极值和标准化需要在截面上处理，行业中性化和市值中性化直接导入一整个dataframe
    """
    # 去除缺失值
    prepared_df = prepared_df.dropna()
    for factor_name in factor_list:
        # 去极值
        prepared_df = prepared_df.groupby('trade_date').apply(winsorize_mad, factor_name=factor_name, n=5).reset_index(drop=True)
        # 标准化
        prepared_df = prepared_df.groupby('trade_date').apply(standardize, factor_name=factor_name).reset_index(drop=True)
        # 中性化,无需分组所以不用apply
        prepared_df =neutralize(prepared_df, factor_name=factor_name, industry_columns=industry_columns)

    return prepared_df

def process_df(factor_list, df,future_n_days_return):
    """
    传入因子列表，原始交易数据，未来n天收益率
    返回一个处理过的DataFrame,已经可以用于检验
    """
    industry_matrix = get_industry_matrix(df)
    industry_columns = list(industry_matrix.columns)
    df,next_period_return = process_return(df,future_n_days_return)
    df = prepared_process(factor_list, df, industry_matrix,next_period_return)
    # 内部循环处理多个因子
    df = process_factor_function(df,factor_list,industry_columns)
    df = df[['ts_code','trade_date','total_mv',next_period_return]+factor_list].sort_values(by=['ts_code','trade_date'],ascending=[True,False])

    return df,next_period_return

if __name__ == "__main__":
    trading_data,output_path = get_trading_data("D:\\Coding_Programs\\Quant\\factor_test\\全部A股上市公司数据.csv",
                                                "D:\\Coding_Programs\\Quant\\factor_test\\")
    processed_df,next_period_return = process_df(['turnover_rate'],trading_data,5)
    processed_df.to_csv(output_path+'processed_data.csv',index=False)


