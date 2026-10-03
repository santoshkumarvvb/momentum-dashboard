import io
import requests
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

st.set_page_config(page_title="Global Stage 2 Momentum Dashboard", page_icon="📈", layout="wide")
st.title("📈 Global Stage 2 Momentum Dashboard")
st.caption("India • USA • Europe/UK • Stage 2 + 52-week-high + multi-period relative-strength momentum • Yahoo Finance")

NIFTY500_URL="https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv"
EUROPE=[("AZN.L","AstraZeneca","UK","GBP","Health Care"),("SHEL.L","Shell","UK","GBP","Energy"),("HSBA.L","HSBC Holdings","UK","GBP","Financials"),("ULVR.L","Unilever","UK","GBP","Consumer Staples"),("REL.L","RELX","UK","GBP","Industrials"),("LSEG.L","London Stock Exchange Group","UK","GBP","Financials"),("RR.L","Rolls-Royce Holdings","UK","GBP","Industrials"),("BA.L","BAE Systems","UK","GBP","Industrials"),("NG.L","National Grid","UK","GBP","Utilities"),("GSK.L","GSK","UK","GBP","Health Care"),("SAP.DE","SAP","Germany","EUR","Technology"),("SIE.DE","Siemens","Germany","EUR","Industrials"),("ALV.DE","Allianz","Germany","EUR","Financials"),("DTE.DE","Deutsche Telekom","Germany","EUR","Communication Services"),("MUV2.DE","Munich Re","Germany","EUR","Financials"),("RHM.DE","Rheinmetall","Germany","EUR","Industrials"),("AIR.PA","Airbus","France","EUR","Industrials"),("MC.PA","LVMH","France","EUR","Consumer Discretionary"),("OR.PA","L'Oreal","France","EUR","Consumer Staples"),("SU.PA","Schneider Electric","France","EUR","Industrials"),("SAF.PA","Safran","France","EUR","Industrials"),("TTE.PA","TotalEnergies","France","EUR","Energy"),("ASML.AS","ASML Holding","Netherlands","EUR","Technology"),("INGA.AS","ING Groep","Netherlands","EUR","Financials"),("PHIA.AS","Philips","Netherlands","EUR","Health Care"),("AD.AS","Ahold Delhaize","Netherlands","EUR","Consumer Staples"),("NOVN.SW","Novartis","Switzerland","CHF","Health Care"),("ROG.SW","Roche","Switzerland","CHF","Health Care"),("NESN.SW","Nestle","Switzerland","CHF","Consumer Staples"),("UBSG.SW","UBS Group","Switzerland","CHF","Financials"),("ABBN.SW","ABB","Switzerland","CHF","Industrials")]

with st.sidebar:
    st.header("Settings")
    market=st.selectbox("Market",["India — Nifty 500","USA — S&P 500","USA — Nasdaq 100","USA — Combined","Europe / UK","Global"])
    st.caption("Data refreshes when you reload the page or change a setting.")
    if st.button("🔄 Refresh market data",use_container_width=True): st.cache_data.clear(); st.rerun()
    only_stage2=st.checkbox("Stage 2 only",False)
    high_filter=st.selectbox("52-week-high filter",["All","Within 25%","Within 10%","Within 5%","Within 2%"])
    min_rsi=st.slider("Minimum RSI(14)",0,80,0,5)
    min_relvol=st.select_slider("Minimum relative volume",options=[0.0,.75,1.0,1.25,1.5,2.0],value=0.0)

@st.cache_data(ttl=86400,show_spinner=False)
def load_india():
    try:
        r=requests.get(NIFTY500_URL,headers={"User-Agent":"Mozilla/5.0","Accept":"text/csv,text/plain,*/*"},timeout=20);r.raise_for_status();raw=pd.read_csv(io.BytesIO(r.content));sc=next(c for c in raw.columns if str(c).strip().lower()=="symbol");cc=next((c for c in raw.columns if "company" in str(c).lower()),sc);sec=next((c for c in raw.columns if "industry" in str(c).lower() or "sector" in str(c).lower()),None)
        return pd.DataFrame({"Ticker":raw[sc].astype(str).str.strip()+".NS","Company":raw[cc].astype(str),"Sector":raw[sec].astype(str) if sec else "","Country":"India","Currency":"INR"}).drop_duplicates("Ticker")
    except Exception:
        raw=pd.read_csv("tickers.csv");out=raw[["Ticker","Company","Sector"]].copy();out["Country"]="India";out["Currency"]="INR";return out
@st.cache_data(ttl=86400,show_spinner=False)
def load_sp500():
    r=pd.read_csv("sp500.csv");o=pd.DataFrame({"Ticker":r["Symbol"].astype(str).str.replace(".","-",regex=False),"Company":r["Security"],"Sector":r["GICS Sector"]});o["Country"]="USA";o["Currency"]="USD";return o.drop_duplicates("Ticker")
@st.cache_data(ttl=86400,show_spinner=False)
def load_nasdaq():
    r=pd.read_csv("nasdaq100.csv");o=pd.DataFrame({"Ticker":r["Ticker"].astype(str),"Company":r["Company"],"Sector":r.get("GICS_Sector","")});o["Country"]="USA";o["Currency"]="USD";return o.drop_duplicates("Ticker")
@st.cache_data(ttl=86400,show_spinner=False)
def load_europe():return pd.DataFrame(EUROPE,columns=["Ticker","Company","Country","Currency","Sector"])
def get_universe(n):
    if n=="India — Nifty 500":return load_india()
    if n=="USA — S&P 500":return load_sp500()
    if n=="USA — Nasdaq 100":return load_nasdaq()
    if n=="USA — Combined":return pd.concat([load_sp500(),load_nasdaq()]).drop_duplicates("Ticker")
    if n=="Europe / UK":return load_europe()
    return pd.concat([load_india(),load_sp500(),load_nasdaq(),load_europe()]).drop_duplicates("Ticker")

def benchmark_for(country):
    if country=="India":return "^NSEI"
    if country=="USA":return "^GSPC"
    if country=="UK":return "^FTSE"
    if country=="Germany":return "^GDAXI"
    if country=="France":return "^FCHI"
    if country=="Netherlands":return "^AEX"
    if country=="Switzerland":return "^SSMI"
    return "^STOXX"

def period_return(s,days):
    s=s.dropna()
    if len(s)<=days:return np.nan
    return (float(s.iloc[-1])/float(s.iloc[-days-1])-1)*100

@st.cache_data(ttl=900,show_spinner=False)
def benchmark_returns(ticker):
    try:
        h=yf.download(ticker,period="18mo",interval="1d",auto_adjust=True,progress=False,threads=False)
        if isinstance(h.columns,pd.MultiIndex):h.columns=h.columns.get_level_values(0)
        c=h["Close"].dropna().astype(float)
        return {63:period_return(c,63),126:period_return(c,126),252:period_return(c,252)}
    except Exception:return {63:np.nan,126:np.nan,252:np.nan}

@st.cache_data(ttl=900,show_spinner=False)
def fetch_one(ticker):
    h=yf.download(ticker,period="18mo",interval="1d",auto_adjust=False,progress=False,threads=False)
    if h is None or h.empty:return None,None
    if isinstance(h.columns,pd.MultiIndex):h.columns=h.columns.get_level_values(0)
    h=h.dropna(subset=["Close"]).copy()
    if len(h)<220:return None,h
    close,high,low,vol=[h[c].astype(float) for c in ["Close","High","Low","Volume"]]
    for n in [20,50,100,150,200]:h[f"EMA{n}"]=close.ewm(span=n,adjust=False).mean()
    d=close.diff();g=d.clip(lower=0);l=-d.clip(upper=0);ag=g.ewm(alpha=1/14,min_periods=14,adjust=False).mean();al=l.ewm(alpha=1/14,min_periods=14,adjust=False).mean();rs=ag/al.replace(0,np.nan);h["RSI14"]=100-(100/(1+rs));h["AvgVol20"]=vol.rolling(20).mean();h["RelVol"]=vol/h["AvgVol20"]
    row=h.iloc[-1];p=float(row["Close"]);e20,e50,e100,e150,e200=[float(row[f"EMA{n}"]) for n in [20,50,100,150,200]];old=float(h["EMA200"].iloc[-21]);hi=float(high.tail(252).max());lo=float(low.tail(252).min());dist=(p/hi-1)*100;rsi=float(row["RSI14"]) if pd.notna(row["RSI14"]) else np.nan;rv=float(row["RelVol"]) if pd.notna(row["RelVol"]) else np.nan
    r3,r6,r12=period_return(close,63),period_return(close,126),period_return(close,252)
    stage=p>e50>e150>e200 and e200>old and p>=1.30*lo and p>=.75*hi;stack=e20>e50>e100>e200;w25,w10,w5,w2=dist>=-25,dist>=-10,dist>=-5,dist>=-2;fresh=p>=hi*.999;vb=bool(pd.notna(rv) and rv>=1.5)
    return {"Ticker":ticker,"Price":p,"EMA20":e20,"EMA50":e50,"EMA100":e100,"EMA150":e150,"EMA200":e200,"EMA Stack":stack,"EMA200 Rising":e200>old,"52W High":hi,"% From 52W High":dist,"Within 25%":w25,"Within 10%":w10,"Within 5%":w5,"Within 2%":w2,"Fresh 52W High":fresh,"RSI14":rsi,"Rel Volume":rv,"Volume Breakout":vb,"Stage 2":stage,"Return 3M %":r3,"Return 6M %":r6,"Return 12M %":r12},h

u=get_universe(market).reset_index(drop=True);st.info(f"Loaded {len(u)} symbols for {market}. Market history is fetched from Yahoo Finance.")
bench_cache={};records=[];histories={}
with st.spinner("Downloading market data and calculating Stage 2 + relative-strength momentum..."):
    prog=st.progress(0);total=max(len(u),1)
    for i,(_,s) in enumerate(u.iterrows(),1):
        t=str(s["Ticker"]).strip()
        try:
            rec,h=fetch_one(t)
            if rec is not None:
                for c in ["Company","Sector","Country","Currency"]:rec[c]=s.get(c,"")
                bt=benchmark_for(rec["Country"])
                if bt not in bench_cache:bench_cache[bt]=benchmark_returns(bt)
                br=bench_cache[bt];rec["Benchmark"]=bt;rec["RS 3M %"]=rec["Return 3M %"]-br[63] if pd.notna(rec["Return 3M %"]) and pd.notna(br[63]) else np.nan;rec["RS 6M %"]=rec["Return 6M %"]-br[126] if pd.notna(rec["Return 6M %"]) and pd.notna(br[126]) else np.nan;rec["RS 12M %"]=rec["Return 12M %"]-br[252] if pd.notna(rec["Return 12M %"]) and pd.notna(br[252]) else np.nan
                records.append(rec);histories[t]=h
        except Exception:pass
        prog.progress(i/total)
    prog.empty()
if not records:st.error("No usable Yahoo Finance data was returned. Please try again later.");st.stop()

df=pd.DataFrame(records);all_df=df.copy()
# Cross-sectional percentile ranks reward stocks outperforming peers without hard-coding return thresholds.
for col,out in [("Return 3M %","3M Rank"),("Return 6M %","6M Rank"),("Return 12M %","12M Rank")]:df[out]=df[col].rank(pct=True)*100
trend_points=np.where(df["Stage 2"],25,0)+np.where(df["EMA Stack"],10,0)
high_points=np.select([df["Within 2%"],df["Within 5%"],df["Within 10%"],df["Within 25%"]],[15,11,7,3],default=0)
volume_points=np.select([df["Rel Volume"]>=1.5,df["Rel Volume"]>=1.25,df["Rel Volume"]>=1.0],[10,7,4],default=0)
rsi_points=np.select([(df["RSI14"]>=60)&(df["RSI14"]<=75),(df["RSI14"]>=55)&(df["RSI14"]<80)],[10,5],default=0)
return_points=(df["3M Rank"].fillna(0)*.08+df["6M Rank"].fillna(0)*.10+df["12M Rank"].fillna(0)*.12)
rs_bonus=np.select([(df["RS 3M %"]>0)&(df["RS 6M %"]>0)&(df["RS 12M %"]>0),(df["RS 3M %"]>0)&(df["RS 6M %"]>0)],[10,5],default=0)
# Normalize to 100 because raw components can total 110.
df["Global Momentum Score"]=np.minimum(100,np.round((trend_points+high_points+volume_points+rsi_points+return_points+rs_bonus)*100/110)).astype(int)
df["Action"]=np.select([(df["Stage 2"])&(df["Within 2%"])&(df["Rel Volume"]>=1.5)&(df["RS 6M %"]>0),(df["Stage 2"])&(df["Within 5%"])&(df["RS 6M %"]>0),df["Stage 2"]],["BREAKOUT","WATCH","STAGE 2"],default="AVOID")
all_df=df.copy()
if only_stage2:df=df[df["Stage 2"]]
fm={"Within 25%":"Within 25%","Within 10%":"Within 10%","Within 5%":"Within 5%","Within 2%":"Within 2%"}
if high_filter in fm:df=df[df[fm[high_filter]]]
if min_rsi:df=df[df["RSI14"]>=min_rsi]
if min_relvol:df=df[df["Rel Volume"]>=min_relvol]
df=df.sort_values(["Global Momentum Score","RS 6M %","% From 52W High"],ascending=[False,False,False]).reset_index(drop=True)

c1,c2,c3,c4,c5=st.columns(5);c1.metric("Stocks scanned",len(all_df));c2.metric("Stage 2",int(all_df["Stage 2"].sum()));c3.metric("Within 5%",int(all_df["Within 5%"].sum()));c4.metric("Within 2%",int(all_df["Within 2%"].sum()));c5.metric("Breakouts",int((all_df["Action"]=="BREAKOUT").sum()))

if market.startswith("USA") or market=="Global":
    st.subheader("🇺🇸 Top USA Momentum")
    usa=all_df[all_df["Country"]=="USA"].sort_values(["Global Momentum Score","RS 6M %"],ascending=False).head(20)
    usa_cols=["Ticker","Company","Sector","Price","Global Momentum Score","Action","Stage 2","% From 52W High","RSI14","Rel Volume","Return 3M %","Return 6M %","Return 12M %","RS 3M %","RS 6M %","RS 12M %"]
    usa_show=usa[usa_cols].copy()
    for c in usa_show.columns:
        if c not in ["Ticker","Company","Sector","Action","Stage 2"]:usa_show[c]=pd.to_numeric(usa_show[c],errors="coerce").round(2)
    st.dataframe(usa_show,use_container_width=True,hide_index=True)
    st.caption("RS columns = stock return minus S&P 500 return over the same period. Ranking is systematic, not a buy recommendation.")

st.subheader(f"🔥 {market} Momentum Ranking")
cols=["Country","Ticker","Company","Sector","Currency","Price","Global Momentum Score","Action","Stage 2","% From 52W High","RSI14","Rel Volume","Return 3M %","Return 6M %","Return 12M %","RS 3M %","RS 6M %","RS 12M %","EMA20","EMA50","EMA100","EMA150","EMA200"]
show=df[cols].copy()
for c in show.columns:
    if c not in ["Country","Ticker","Company","Sector","Currency","Action","Stage 2"]:show[c]=pd.to_numeric(show[c],errors="coerce").round(2)
st.dataframe(show,use_container_width=True,hide_index=True);st.download_button("Download current ranking",data=df.to_csv(index=False).encode("utf-8"),file_name="global_momentum_ranking.csv",mime="text/csv")

if not df.empty:
    st.subheader("📊 Stock Detail");selected=st.selectbox("Select stock",df["Ticker"].tolist());row=df[df["Ticker"]==selected].iloc[0];sym={"INR":"₹","USD":"$","GBP":"£","EUR":"€","CHF":"CHF "}.get(row["Currency"],"");a,b,c,d=st.columns(4);a.metric("Price",f"{sym}{row['Price']:.2f}");b.metric("Global Momentum",f"{int(row['Global Momentum Score'])}/100");c.metric("6M Return",f"{row['Return 6M %']:.1f}%");d.metric("6M Relative Strength",f"{row['RS 6M %']:+.1f}%");st.line_chart(histories[selected].tail(180)[["Close","EMA20","EMA50","EMA100","EMA200"]],use_container_width=True);st.write(f"**Signal:** {row['Action']} | **Stage 2:** {'Yes' if row['Stage 2'] else 'No'} | **52W distance:** {row['% From 52W High']:.2f}% | **RSI:** {row['RSI14']:.1f} | **Relative volume:** {row['Rel Volume']:.2f}× | **Benchmark:** {row['Benchmark']}")

st.caption("Systematic momentum screening only: Stage 2, 52-week-high proximity, volume, RSI, 3/6/12-month returns and benchmark-relative strength. Not investment advice.")