#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""中证500 V1/B 周频指数研究（无ETF）。适用于 GitHub Actions，2016年起收集日度，2021年起样本外。
V1=Δ[中国10Y每周最后有效值 - 美国10Y周均值]
B=Δ[中国5Y每周最后有效值 - 美国5Y周均值]
其他两因子：100·Δln(USD/CNY周均), 100·Δln(美国广义美元指数周均)
信号：星期五结束的周w数据，一般在下周二（美国H10发布后）可见；为容纳节假日，星期四及之后交易。
预测：周w的宏观信号，预测周w+h的中证500价格指数周四/周五开盘至下一可交易周同规则开盘的单周收益。
评估：仅用中证500价格指数计算训练标签、预测、方向准确率和理论择时曲线；不下载ETF。指数不可直接交易。
严格训练：在评估信号周w时，仅训练目标周<=w-1的已经完结的收益。
注意：历史H10修订、宏观数据实际公布时间、指数历史行情准确性及交易可行性仍需审计。
"""
import argparse
from io import StringIO
from pathlib import Path
import json, time, sys, os, traceback
import numpy as np
import pandas as pd
import requests
import statsmodels.api as sm

REPO = 'https://raw.githubusercontent.com/zhesunstudentutseduau-afk/macro-open-data/main/data/'
ROOT = Path('research/weekly_index_only')
START='20160101'; END=pd.Timestamp.today().strftime('%Y%m%d')
HORIZONS=[1,2,4,8]
START_EVAL='2021-03-05'
MIN_TRAIN=130
TRADE_COST=0.001    # 仅用于理论指数择时代理的换仓成本敏感性假设，不代表真实可执行成本


def download_csv(url, tries=3):
    last=None
    for i in range(tries):
        try:
            r=requests.get(url,timeout=45,headers={'User-Agent':'Mozilla/5.0 (research weekly macro index)'});
            r.raise_for_status()
            df=pd.read_csv(StringIO(r.content.decode('utf-8-sig')))
            if len(df)<80: raise RuntimeError(f'Data too short: {url}, {len(df)} rows')
            return df
        except Exception as exc:
            last=exc
            print('download attempt',i+1,'failed',url,str(exc)[:250],flush=True)
            time.sleep(2*(i+1))
    raise RuntimeError(f'Download failed: {url}: {last}')


def normalize_price(raw, tag):
    df=raw.copy(); df.columns=[str(c).strip() for c in df.columns]
    ren={'日期':'date','date':'date','开盘':'open','open':'open','收盘':'close','close':'close',
         '成交量':'volume','volume':'volume'}
    df=df.rename(columns={k:v for k,v in ren.items() if k in df.columns})
    for c in ['date','open','close']:
        if c not in df: raise ValueError(f'{tag}: missing {c}: {list(df.columns)}')
    df['date']=pd.to_datetime(df['date'],errors='coerce')
    for c in ['open','close']:
        df[c]=pd.to_numeric(df[c],errors='coerce')
    df=df[['date','open','close']].dropna().sort_values('date').drop_duplicates('date',keep='last')
    df=df[(df.date.dt.weekday<5)&(df.open>0)&(df.close>0)]
    if len(df)<1000: raise ValueError(f'{tag} insufficient trading dates: {len(df)}')
    df.to_csv(ROOT / f'{tag}_daily.csv',index=False,encoding='utf-8-sig')
    return df


def validate_market_coverage(data, tag):
    # Never silently use short/stale market data: that would distort the walk-forward sample.
    first, last = data.date.min(), data.date.max()
    if first > pd.Timestamp('2017-01-01'):
        raise ValueError(f'{tag} starts too late for 2016 baseline: {first.date()}')
    if last < pd.Timestamp.today().normalize() - pd.Timedelta(days=21):
        raise ValueError(f'{tag} stale: last date {last.date()}')
    print(tag, 'COVERAGE_OK', str(first.date()), str(last.date()), 'n=', len(data), flush=True)
    return data


def _baostock_worker(code, adjustflag, output):
    """Child process: BaoStock has had reports of unbounded TCP reads."""
    import baostock as bs
    import socket
    socket.setdefaulttimeout(25)
    try:
        print('BAOSTOCK child connecting',code,flush=True)
        lg=bs.login()
        if lg.error_code!='0':
            raise RuntimeError(f'Login failed: {lg.error_code} {lg.error_msg}')
        try:
            q=bs.query_history_k_data_plus(code,'date,open,close',
                start_date='2016-01-01',end_date=pd.Timestamp.today().strftime('%Y-%m-%d'),
                frequency='d',adjustflag=adjustflag)
            if q.error_code!='0':
                raise RuntimeError(f'Query error: {q.error_code} {q.error_msg}')
            rows=[]
            while q.next():rows.append(q.get_row_data())
            pd.DataFrame(rows,columns=q.fields).to_csv(output,index=False)
            print('BAOSTOCK child complete',code,len(rows),flush=True)
        finally:bs.logout()
    except Exception as exc:
        Path(output+'.error').write_text(repr(exc),encoding='utf-8')
        raise


def baostock_daily(code,adjustflag='3'):
    import multiprocessing as mp
    # Terminate a stuck provider process instead of hanging the entire weekly job.
    target=ROOT/('baostock_'+code.replace('.','_')+'_'+adjustflag+'.tmp.csv')
    ctx=mp.get_context('fork')  # GitHub-hosted runner is ubuntu-latest
    proc=ctx.Process(target=_baostock_worker,args=(code,adjustflag,str(target)))
    proc.start();proc.join(timeout=75)
    if proc.is_alive():
        proc.terminate();proc.join(timeout=5)
        raise RuntimeError(f'Baostock timeout after 75s: {code}')
    if proc.exitcode!=0 or not target.exists():
        err=Path(str(target)+'.error')
        raise RuntimeError(f'Baostock failed: {code}, exit={proc.exitcode}, details=' +
            (err.read_text(encoding='utf-8') if err.exists() else 'no details'))
    result=pd.read_csv(target)
    target.unlink(missing_ok=True)
    return result


def read_market():
    import akshare as ak

    # We must use CSI500 PRICE index (000905), not total-return index or CSI300.
    # Two legacy Eastmoney APIs failed on the initial 2026-10-31 Actions run.
    # Fallbacks query independent providers; all must pass the same quality checks.
    def fallback_index():
        errors=[]
        providers=[
            ('CSI official index_zh_a_hist_csindex 000905',
             lambda:ak.stock_zh_index_hist_csindex(symbol='000905',start_date=START,end_date=END)),
            ('Sina stock_zh_index_daily sh000905',
             lambda:ak.stock_zh_index_daily(symbol='sh000905')),
            ('baostock sh.000905',lambda:baostock_daily('sh.000905')),
            ('Eastmoney stock_zh_index_daily_em csi000905',
             lambda:ak.stock_zh_index_daily_em(symbol='csi000905',start_date=START,end_date=END)),
            ('Eastmoney index_zh_a_hist 000905',
             lambda:ak.index_zh_a_hist(symbol='000905',period='daily',start_date=START,end_date=END)),
        ]
        for label,fn in providers:
            try:
                raw=fn()
                print('CSI500 source candidate',label,'rows',len(raw),flush=True)
                ret=validate_market_coverage(normalize_price(raw,'csi500_index'), 'CSI500 '+label)
                (ROOT/'csi500_source.txt').write_text(label,encoding='utf-8')
                return ret
            except Exception as exc:
                errors.append((label,str(exc)[:500]))
                print('CSI500 source rejected:',label,str(exc)[:500],flush=True)
        raise RuntimeError('All independent CSI500 index sources failed: '+repr(errors))

    idx=fallback_index()
    # Explicit daily sanity checks; do not replace suspicious data with synthetic values.
    daily=idx[['date','open','close']].copy()
    daily['open_to_open_pct']=daily.open.pct_change()*100
    daily['close_to_close_pct']=daily.close.pct_change()*100
    audit={
      'index_symbol':'000905', 'index_variant':'price index (not gross/net total return)',
      'provider':(ROOT/'csi500_source.txt').read_text(encoding='utf-8'),
      'first_date':str(daily.date.min().date()),'last_date':str(daily.date.max().date()),
      'n_trading_days':len(daily),
      'n_open_changes_above_20pct':int((daily.open_to_open_pct.abs()>20).sum()),
      'n_close_changes_above_20pct':int((daily.close_to_close_pct.abs()>20).sum()),
      'largest_abs_daily_open_change_pct':round(float(daily.open_to_open_pct.abs().max()),4),
      'largest_abs_daily_close_change_pct':round(float(daily.close_to_close_pct.abs().max()),4),
    }
    (ROOT/'index_price_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf8')
    if audit['n_open_changes_above_20pct'] or audit['n_close_changes_above_20pct']:
        raise RuntimeError('CSI500 index history has >20% day-to-day price jumps; manual audit needed: '+str(audit))
    print('INDEX_PRICE_AUDIT',audit,flush=True)
    return idx





def read_bonds_latest():
    """Prefer refresh with AKShare at GitHub run; fallback to previously archived Github daily.
    Failed refresh does not invent 2026-10 observations.
    """
    import akshare as ak
    cols={'中国国债收益率2年':'CN_2Y_pct', '中国国债收益率5年':'CN_5Y_pct',
          '中国国债收益率10年':'CN_10Y_pct', '美国国债收益率2年':'US_2Y_pct',
          '美国国债收益率5年':'US_5Y_pct', '美国国债收益率10年':'US_10Y_pct'}
    try:
        raw=ak.bond_zh_us_rate(start_date=START)
        if not set(cols).union({'日期'}).issubset(raw.columns):
            raise ValueError('Bond AKShare schema change: '+str(raw.columns.tolist()))
        b=raw[['日期',*cols]].rename(columns=cols).copy()
        b['日期']=pd.to_datetime(b['日期'],errors='coerce')
        for col in cols.values():b[col]=pd.to_numeric(b[col],errors='coerce')
        b=b.dropna(subset=['日期']).sort_values('日期').drop_duplicates('日期',keep='last')
        if len(b)<1800:raise ValueError('Bond daily refresh too short')
        b['日期']=b['日期'].dt.strftime('%Y-%m-%d')
        print('Live bond refresh succeeded through',b['日期'].max(),flush=True)
    except Exception as err:
        print('WARNING: live bond refresh failed, fallback to existing public Github CSV',err,flush=True)
        b=download_csv(REPO+'china_us_yields_daily.csv')
    b.to_csv(ROOT/'bonds_daily_source.csv',index=False,encoding='utf-8-sig')
    latest=pd.to_datetime(b['日期'],errors='coerce').max()
    age=(pd.Timestamp.today().normalize()-latest.normalize()).days
    print('Bond latest',latest.date(),'age_days',age,flush=True)
    if age>21:print('WARNING: bond data are stale: results will end early; do NOT claim current signals',flush=True)
    return b


def read_fred(series):
    """H.10 daily Fed data: direct Board historical tables first, FRED CSV fallback.
    DEXCHUS = CNY per USD; DTWEXBGS = 2006-base nominal broad USD index.
    No proxy substitutions, missing-value interpolation or silent stale data.
    """
    from bs4 import BeautifulSoup
    import re
    fed_pages={
        'DEXCHUS':'https://www.federalreserve.gov/releases/h10/hist/dat00_ch.htm',
        'DTWEXBGS':'https://www.federalreserve.gov/releases/h10/summary/jrxwtfb_nb.htm',
    }
    if series not in fed_pages:raise ValueError('Unrecognized Fed H10 series '+series)

    def normalize(raw):
        if not {'date',series}.issubset(raw.columns):
            raise ValueError(series+' missing required date/value fields '+str(raw.columns.tolist()))
        res=raw[['date',series]].copy()
        res['date']=pd.to_datetime(res['date'],errors='coerce')
        res[series]=pd.to_numeric(res[series],errors='coerce')
        res=res.dropna(subset=['date']).drop_duplicates('date',keep='last').sort_values('date')
        res=res[res.date>=pd.Timestamp('2016-01-01')]
        valid=res.dropna(subset=[series])
        if len(valid)<1600:raise RuntimeError(f'{series}: too few valid daily rows: {len(valid)}')
        if valid.date.min()>pd.Timestamp('2017-01-01'):
            raise RuntimeError(f'{series}: first available date too late {valid.date.min()}')
        last=valid.date.max()
        age=(pd.Timestamp.today().normalize()-last).days
        if age>35:raise RuntimeError(f'{series}: stale last non-missing observation: {last.date()} age {age}d')
        if series=='DEXCHUS' and not valid[series].between(3,12).all():
            raise RuntimeError('CNY per USD outside broad sanity range, check units')
        if series=='DTWEXBGS' and not valid[series].between(50,250).all():
            raise RuntimeError('Broad USD index outside broad sanity range, check units')
        return res

    errors=[]
    # The original FRED fredgraph.csv timed out in GitHub Actions #2.
    # Use the Federal Reserve Board's publicly displayed original H.10 daily history.
    try:
        url=fed_pages[series]
        r=requests.get(url,timeout=(12,40),headers={'User-Agent':'Mozilla/5.0 (academic macro data research)'})
        r.raise_for_status()
        soup=BeautifulSoup(r.text,'html.parser')
        rows=[]
        for tr in soup.find_all('tr'):
            cells=[c.get_text(' ',strip=True) for c in tr.find_all(['td','th'],recursive=False)]
            if len(cells)>=2 and re.fullmatch(r'\d{1,2}-[A-Za-z]{3}-\d{2}',cells[0]):
                rows.append((cells[0],cells[1]))
        if len(rows)<1600:raise RuntimeError(f'Fed H10 history parser found only {len(rows)} rows')
        # Fixed format prevents an ambiguous day/month interpretation.
        tmp=pd.DataFrame(rows,columns=['raw_date',series])
        tmp['date']=pd.to_datetime(tmp.pop('raw_date'),format='%d-%b-%y',errors='coerce')
        out=normalize(tmp)
        source='Federal Reserve Board H.10 historical daily HTML: '+url
    except Exception as exc:
        errors.append(('Federal Reserve Board H.10',str(exc)[:300]))
        print('WARN: Fed Board H.10 unavailable, trying original FRED CSV',series,str(exc)[:300],flush=True)
        try:
            u=f'https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}&cosd=2016-01-01'
            df=download_csv(u,tries=1)
            vcol=series if series in df else df.columns[-1]
            dcol='observation_date' if 'observation_date' in df else df.columns[0]
            out=normalize(pd.DataFrame({'date':df[dcol],series:df[vcol]}))
            source='FRED original fredgraph.csv: '+u
        except Exception as exc2:
            errors.append(('FRED CSV',str(exc2)[:300]))
            raise RuntimeError(f'H10 daily data unavailable for {series}: '+repr(errors)) from exc2
    out.to_csv(ROOT/f'{series}_fred_daily.csv',index=False,encoding='utf-8-sig')
    (ROOT/f'{series}_source.txt').write_text(source,encoding='utf-8')
    print('H10_SERIES_OK',series,'source=',source,
          'valid_n=',out[series].notna().sum(),
          'first=',out.date.min().date(),'last=',out.loc[out[series].notna(),'date'].max().date(),flush=True)
    return out


def make_week(dates):
    return dates.dt.to_period('W-FRI').dt.end_time.dt.normalize()


def aggregate_week_last(df, field, output, week_axis):
    d=df[['date',field]].copy().dropna(); d['week']=make_week(d.date)
    d=d[(d['date'].dt.weekday<5)&(d['date']>=pd.Timestamp('2016-01-01'))]
    # yields: last effective observation in week, not assuming Friday exists
    return d.groupby('week').tail(1).set_index('week')[field].rename(output).reindex(week_axis)


def aggregate_week_mean(df,field,output,week_axis):
    d=df[['date',field]].copy().dropna();d['week']=make_week(d.date)
    d=d[(d.date.dt.weekday<5)&(d.date>=pd.Timestamp('2016-01-01'))]
    count=d.groupby('week')[field].count()
    s=d.groupby('week')[field].mean()
    # 2 observations minimum for full ordinary weeks; short holiday weeks remain missing to avoid invented values
    return s.where(count>=2).rename(output).reindex(week_axis)


def build_features(bonds, fx, dollar):
    if '日期' not in bonds: raise ValueError('Bond daily schema changed')
    b=bonds.copy(); b['date']=pd.to_datetime(b['日期'],errors='coerce')
    req=['中国国债收益率5年','中国国债收益率10年','美国国债收益率5年','美国国债收益率10年']
    # Github保存的CSV已经是英文规范化字段
    mapping={
        'CN_5Y_pct':'CN_5Y_pct','CN_10Y_pct':'CN_10Y_pct',
        'US_5Y_pct':'US_5Y_pct','US_10Y_pct':'US_10Y_pct',
    }
    missing=[c for c in mapping if c not in b]
    if missing:raise ValueError('GitHub daily bond column changed:'+str(missing))
    for c in mapping:b[c]=pd.to_numeric(b[c],errors='coerce')
    weeks=pd.date_range('2016-01-08',pd.Timestamp.today()+pd.Timedelta(days=30),freq='W-FRI')
    m=pd.DataFrame(index=weeks)
    for maturity in (5,10):
        m[f'CN_{maturity}Y_last']=aggregate_week_last(b,f'CN_{maturity}Y_pct',f'CN_{maturity}Y_last',weeks)
        m[f'US_{maturity}Y_avg']=aggregate_week_mean(b,f'US_{maturity}Y_pct',f'US_{maturity}Y_avg',weeks)
    m['CNY_avg']=aggregate_week_mean(fx,'DEXCHUS','CNY_avg',weeks)
    m['Dollar_avg']=aggregate_week_mean(dollar,'DTWEXBGS','Dollar_avg',weeks)
    for maturity in (5,10):
        m[f'bond_spread_{maturity}Y']=m[f'CN_{maturity}Y_last']-m[f'US_{maturity}Y_avg']
        m[f'd_spread_{maturity}Y']=m[f'bond_spread_{maturity}Y'].diff()
    m['fx_log_change']=100*np.log(m.CNY_avg).diff()
    m['dollar_log_change']=100*np.log(m.Dollar_avg).diff()
    # 删除不完整的宏观周但不向前填值，不删除时间轴
    m.index.name='week_end'
    m.to_csv(ROOT/'weekly_macro_factors.csv',encoding='utf-8-sig')
    return m


def price_weekly(index_df, weeks):
    """Index-only weekly theoretical returns. Never represents a tradable execution price."""
    d=index_df[['date','open']].copy()
    d['week_end']=make_week(d.date)
    d=d[d.date.dt.weekday>=3].sort_values('date').groupby('week_end').head(1).set_index('week_end').sort_index()
    d.columns=['entry_date_index','entry_open_index']
    aligned=d.reindex(weeks).copy()
    aligned.index.name='week_end'
    prices=aligned['entry_open_index']
    valid=prices.dropna()
    returns=[]; holding_days=[]
    for w in aligned.index:
        if pd.isna(prices.loc[w]):
            returns.append(np.nan); holding_days.append(np.nan); continue
        subsequent=valid[valid.index>w]
        if subsequent.empty:
            returns.append(np.nan); holding_days.append(np.nan); continue
        returns.append(float(subsequent.iloc[0]/prices.loc[w]-1))
        holding_days.append(int((subsequent.index[0]-w).days))
    aligned['return_index']=returns
    aligned['holding_calendar_days']=holding_days
    aligned.to_csv(ROOT/'weekly_index_anchors_and_returns.csv',encoding='utf-8-sig')
    return aligned


def make_regressors(m,name):
    factor=f'd_spread_{10 if name=="V1" else 5}Y'
    return np.column_stack([m[factor].to_numpy(float),m.fx_log_change.to_numpy(float),m.dollar_log_change.to_numpy(float)])


def evaluate(week_axis, factors, prices, model, horizon, eval_start=START_EVAL):
    x=make_regressors(factors,model)
    y=prices.return_index.to_numpy(float)
    weeks=list(week_axis)
    records=[]
    prev_w=0
    for k in range(len(weeks)):
        if weeks[k]<pd.Timestamp(eval_start):continue
        t=k-horizon
        if t<1 or not np.isfinite(x[t]).all():continue
        if not np.isfinite(y[k]) or pd.isna(prices.iloc[k]['entry_date_index']):continue
        # At signal week t, only labels from target weeks <= t-1 may be used.
        eligible=np.arange(1,max(1,t-horizon),dtype=int)
        eligible=eligible[(eligible+horizon<len(y))]
        eligible=eligible[np.isfinite(x[eligible]).all(axis=1)&np.isfinite(y[eligible+horizon])]
        if len(eligible)<MIN_TRAIN:continue
        fit=sm.OLS(y[eligible+horizon],sm.add_constant(x[eligible],has_constant='add')).fit()
        pred=float(fit.predict(np.r_[1,x[t]].reshape(1,-1))[0])
        weight=int(pred>0)
        cost=TRADE_COST*abs(weight-prev_w)
        records.append({
          'model':model,'horizon_weeks':horizon,'week_end':str(weeks[k].date()),
          'signal_week':str(weeks[t].date()),'index_reference_date':str(prices.iloc[k]['entry_date_index'].date()),
          'train_n':len(eligible),'prediction_index_pct':100*pred,
          'actual_index_pct':100*y[k], 'weight':weight,
          'hypothetical_switch_cost_pct':100*cost,
          'theoretical_index_strategy_before_cost_pct':100*weight*y[k],
          'theoretical_index_strategy_after_cost_pct':100*(weight*y[k]-cost),
          'holding_calendar_days':prices.iloc[k]['holding_calendar_days'],
          'beta_spread':fit.params[1],'beta_fx':fit.params[2],
          'beta_dollar':fit.params[3],'beta_intercept':fit.params[0],
          'train_r2':fit.rsquared})
        prev_w=weight
    return pd.DataFrame(records)


def metrics(df):
    pure=df.theoretical_index_strategy_before_cost_pct.to_numpy(float)/100
    r=df.theoretical_index_strategy_after_cost_pct.to_numpy(float)/100
    benchmark=df.actual_index_pct.to_numpy(float)/100
    accuracy=float(np.mean((df.prediction_index_pct.to_numpy(float)>0)==(df.actual_index_pct.to_numpy(float)>0)))
    nav=np.cumprod(1+r)
    nav_pure=np.cumprod(1+pure)
    bench=np.cumprod(1+benchmark)
    dd=float((nav/np.maximum.accumulate(np.r_[1.,nav])[1:]-1).min())
    fit=sm.OLS(r,sm.add_constant(benchmark)).fit(cov_type='HAC',cov_kwds={'maxlags':4})
    return {'model':df.model.iloc[0], 'horizon_weeks':int(df.horizon_weeks.iloc[0]),
      'first_trade':df.week_end.iloc[0], 'last_trade':df.week_end.iloc[-1],
      'n':len(df),'train_min':int(df.train_n.min()),'train_max':int(df.train_n.max()),
      'theoretical_index_strategy_gross_pct':100*(nav_pure[-1]-1),
      'theoretical_index_strategy_after_assumed_cost_pct':100*(nav[-1]-1),
      'index_price_buyhold_pct':100*(bench[-1]-1),
      'annualized_proxy_return_pct':100*((nav[-1]**(52/len(r)))-1),
      'annualized_proxy_volatility_pct':100*np.std(r,ddof=1)*np.sqrt(52),
      'max_proxy_drawdown_pct':100*dd,'accuracy_pct':100*accuracy,
      'hypothetical_index_exposure_pct':100*df.weight.mean(),
      'n_switches':int((df.hypothetical_switch_cost_pct>0).sum()),
      'alpha_annual_proxy_pct':100*52*fit.params[0],
      'alpha_pvalue_HAC4':float(fit.pvalues[0]),
      'beta_market':float(fit.params[1]),
      'notes':'Index-only theoretical timing study; price index excludes dividends; no investable P&L, not ETF or stock execution.'}


def last_model_predictions(weeks,fac,pr,model,h):
    x=make_regressors(fac,model);y=pr.return_index.to_numpy(float)
    ans=[]
    for t in range(len(weeks)-1,-1,-1):
        if not np.isfinite(x[t]).all():continue
        # Avoid signals based on a Friday whose Fed H10 index is not released yet.
        # End of week Fri, Fed H10 releases Mon/Tue Eastern, conservatively only after the next Thursday in China.
        if pd.Timestamp.today().normalize() < weeks[t] + pd.Timedelta(days=6):continue
        k=t+h
        candidates=np.arange(1,max(1,t-h),dtype=int)
        candidates=candidates[(candidates+h<len(y))]
        candidates=candidates[np.isfinite(x[candidates]).all(axis=1)&np.isfinite(y[candidates+h])]
        if len(candidates)<MIN_TRAIN:continue
        fit=sm.OLS(y[candidates+h],sm.add_constant(x[candidates],has_constant='add')).fit()
        pred=float(fit.predict(np.r_[1,x[t]].reshape(1,-1))[0])
        ans.append({'asof_utc':pd.Timestamp.now(tz='UTC').isoformat(),'model':model,'horizon_weeks':h,
            'signal_week':str(weeks[t].date()),'target_week_end':str(weeks[k].date()) if k<len(weeks) else '',
            'target_thursday_estimate':str((weeks[t]+pd.Timedelta(days=7*h-1)).date()),
            'predicted_index_return_pct':100*pred,'suggested_weight':int(pred>0),'train_n':len(candidates),
            'status':'backfilled research signal; not historical registered forecast'})
        break
    return ans


def quality(fac,prices,idx,fx,dol):
    d={'research_design':'CSI500 price index ONLY; no ETF data and no real-world tradable P&L',
       'raw_bond_source':REPO+'china_us_yields_daily.csv',
       'fed_h10_series':['DEXCHUS','DTWEXBGS'],
       'fed_h10_sources':{k:(ROOT/f'{k}_source.txt').read_text(encoding='utf-8') if (ROOT/f'{k}_source.txt').exists() else 'unknown' for k in ['DEXCHUS','DTWEXBGS']},
       'index_price_source':(ROOT/'csi500_source.txt').read_text(encoding='utf-8') if (ROOT/'csi500_source.txt').exists() else 'unknown',
       'monthly_models':'NOT modified',
       'week_definition':'Week ending Friday (W-FRI); China last available yield, US yield average, FX mean, broad dollar mean',
       'entry_convention':'Index anchor: first Chinese market open Thursday or Friday of target week; next eligible week open as return endpoint; not executable index trading',
       'data_release_assumption':'US H.10 publication delayed on US holidays; Thursday China reference point is a conservative heuristic, NOT point-in-time verified',
       'tplus_h_convention':'factor week t -> index week t+h opening-to-next-week opening SINGLE-period return, NOT h-week compounded cumulative',
       'n_weeks_macro_V1':int(fac[['d_spread_10Y','fx_log_change','dollar_log_change']].notna().all(axis=1).sum()),
       'n_weeks_macro_B':int(fac[['d_spread_5Y','fx_log_change','dollar_log_change']].notna().all(axis=1).sum()),
       'n_index_weekly_anchors':int(prices.entry_open_index.notna().sum()),
       'n_complete_weekly_index_returns':int(prices.return_index.notna().sum()),
       'date_ranges':{'idx':[str(idx.date.min().date()),str(idx.date.max().date())],
                      'fx':[str(fx.date.min().date()),str(fx.date.max().date())],
                      'dollar':[str(dol.date.min().date()),str(dol.date.max().date())]},
       'warning':['CSI500 PRICE index (000905) is NOT a directly tradable asset and does not include dividends reinvested',
                  'Backtested strategy returns are theoretical index exposure only, NOT realizable fund returns',
                  'Hypothetical per-switch cost 0.1% is NOT established execution friction',
                  'Historical H.10 data may reflect later corrections; actual vintage not yet audited',
                  'Historical real-time release of all factor observations is not fully checked',
                  'Holiday-week gaps cause periods longer than 7 days; 52-week annualization approximates elapsed time']}
    d['last_bond_date']=str(pd.to_datetime(pd.read_csv(ROOT/'bonds_daily_source.csv')['日期']).max().date()) if (ROOT/'bonds_daily_source.csv').exists() else 'missing'
    (ROOT/'data_quality.json').write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8')
    return d


def self_test():
    np.random.seed(37)
    weeks=pd.date_range('2016-01-08',periods=570,freq='W-FRI')
    fac=pd.DataFrame(index=weeks)
    for v in ['d_spread_10Y','d_spread_5Y','fx_log_change','dollar_log_change']:
        fac[v]=np.random.normal(0,.3,len(weeks))
    pr=pd.DataFrame(index=weeks)
    pr['entry_date_index']=weeks-pd.Timedelta(days=1)
    pr['holding_calendar_days']=7
    pr['return_index']=.01*fac['d_spread_10Y'].shift(1)+.002*np.random.randn(len(weeks))
    for h in HORIZONS:
        for name in ['V1','B']:
            r=evaluate(weeks,fac,pr,name,h)
            assert len(r)>100,(name,h,len(r))
            assert r.train_n.min()>=MIN_TRAIN
            assert r[['prediction_index_pct','theoretical_index_strategy_after_cost_pct']].notna().all().all()
            assert ((r.week_end.astype('datetime64[ns]')-r.signal_week.astype('datetime64[ns]')).dt.days==7*h).all()
            _=metrics(r)
    original=evaluate(weeks,fac,pr,'V1',2)
    corrupted=pr.copy()
    corrupted.loc[corrupted.index>=pd.Timestamp('2025-01-01'),'return_index']+=.6
    changed=evaluate(weeks,fac,corrupted,'V1',2)
    a=original[original.week_end<'2024-01-01'].prediction_index_pct.to_numpy()
    b=changed[changed.week_end<'2024-01-01'].prediction_index_pct.to_numpy()
    assert np.array_equal(a,b),'LOOKAHEAD ERROR: future labels changed past predictions'
    # Price-grid handling across an absent trading week.
    d=pd.DataFrame({'date':pd.to_datetime(['2024-04-04','2024-04-18','2024-04-25']),
                    'open':[5000.,5050.,5100.]})
    test_weeks=pd.date_range('2024-04-05','2024-04-26',freq='W-FRI')
    saved=ROOT
    try:
        ROOT.mkdir(parents=True,exist_ok=True)
        p=price_weekly(d,test_weeks)
        assert np.isnan(p.loc[pd.Timestamp('2024-04-12'),'return_index'])
        assert abs(p.loc[pd.Timestamp('2024-04-05'),'return_index']-0.01)<1e-10
        assert p.loc[pd.Timestamp('2024-04-05'),'holding_calendar_days']==14
    finally:
        (ROOT/'weekly_index_anchors_and_returns.csv').unlink(missing_ok=True)
    print('SELFTEST_OK: 8 index-only OLS models, future-leakage test, missing-week anchor test. Synthetic data only.')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--self-test',action='store_true');args=ap.parse_args()
    if args.self_test:self_test();return
    ROOT.mkdir(parents=True,exist_ok=True)
    print('Step 1: download actual CSI500 index and macro history (NO ETF)',flush=True)
    bonds=read_bonds_latest()
    idx=read_market()
    fx=read_fred('DEXCHUS');dol=read_fred('DTWEXBGS')
    fac=build_features(bonds,fx,dol)
    prices=price_weekly(idx,fac.index)
    dq=quality(fac,prices,idx,fx,dol)
    print('Data quality:',json.dumps(dq['date_ranges'],ensure_ascii=False),flush=True)
    print('Step 2: expanding-window OLS and INDEX-only theoretical evaluation',flush=True)
    results=[];summ=[];nextsignal=[]
    for model in ['V1','B']:
        for h in HORIZONS:
            rec=evaluate(fac.index,fac,prices,model,h)
            if len(rec)<100:raise RuntimeError(f'{model},h{h}: only {len(rec)} valid predictions; refuse to report')
            results.append(rec)
            m=metrics(rec);summ.append(m)
            print(model,'h',h,'n',len(rec),'index proxy costed%',
                  round(m['theoretical_index_strategy_after_assumed_cost_pct'],2),
                  'index price buyhold%',round(m['index_price_buyhold_pct'],2),
                  'direction accuracy%',round(m['accuracy_pct'],2),flush=True)
            nextsignal+=last_model_predictions(fac.index,fac,prices,model,h)
    summary=pd.DataFrame(summ)
    records=pd.concat(results,ignore_index=True)
    records.to_csv(ROOT/'weekly_v1b_index_only_predictions_backtest.csv',index=False,encoding='utf-8-sig')
    summary.to_csv(ROOT/'weekly_v1b_index_only_summary.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(nextsignal).to_csv(ROOT/'latest_signal_research_only.csv',index=False,encoding='utf-8-sig')
    report='''# 中证500价格指数周频 V1/B 探索性研究（指数版，自动生成）

**这是一份指数预测与理论择时评估，不代表可直接执行的基金交易收益。**

- 预测标的：中证500**价格指数 000905**，非ETF、非全收益指数。整个运行链路不读取ETF数据。
- 8组研究配置：V1/B分别预测相隔t+1/t+2/t+4/t+8周的**目标周单周**指数开盘至下周开盘收益，不是持有h周累计收益。
- 因子定义：原有10Y/5Y中美国债利差变动、美元兑人民币周均变化、广义美元指数周均变化；训练从2016年扩展，评估期自2021年3月。
- 目标周选取周四或周五第一交易日的指数开盘为参考点，下一可交易周同规则开盘为终点。指数本身不能直接交易，此开盘点位只用于定义研究标签。
- 理论择时曲线分别提供不扣费用和每次仓位变化假设0.1%成本两套结果；后者**绝非真实ETF净收益**。
- 仍有H.10历史修订、发布日期、交易日缺口、指数多来源交叉复核等未决问题。

|模型|预测期(周)|有效周|理论指数择时(未扣费)%|理论指数择时(假设扣费)%|指数价格买入持有%|方向准确率%|理论最大回撤%|HAC p值|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
'''
    for m in summ:
        report+=f"|{m['model']}|{m['horizon_weeks']}|{m['n']}|{m['theoretical_index_strategy_gross_pct']:.2f}|{m['theoretical_index_strategy_after_assumed_cost_pct']:.2f}|{m['index_price_buyhold_pct']:.2f}|{m['accuracy_pct']:.2f}|{m['max_proxy_drawdown_pct']:.2f}|{m['alpha_pvalue_HAC4']:.3g}|\n"
    report+='''
## 数据与研究边界

- 历史数据及来源：`csi500_index_daily.csv`、`csi500_source.txt`、`index_price_audit.json`、`data_quality.json`。
- 模型输出：`weekly_v1b_index_only_predictions_backtest.csv`、`weekly_v1b_index_only_summary.csv`。
- 指数价格收益不含成分股现金分红再投资；未来如需计算真实可交易收益，必须另行验证交易工具、费用及跟踪偏离，不应将本报告理论收益直接当作投资回报。
- 对周度汇率、美元指数使用历史已修订数据的回测仍可能存在修订信息问题；须后续补充逐时点可得性审计。
- 多种窗口、模型周期间比较涉及多重检验，单独看到漂亮的准确率不构成预测有效性证据。
- 原有月度模型和历史ETF研究输出不在本次指数版内被覆盖或删除。
'''
    (ROOT/'research_report.md').write_text(report,encoding='utf-8')
    print('OK: generated INDEX-only results in',ROOT,flush=True)


if __name__=='__main__': main()
