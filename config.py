import os
import yaml

cfg = {
    "project": {"data_dir": "data", "artifacts_dir": "artifacts", "random_state": 42},
    "universe": {
        "tickers": ["AAPL","MSFT","AMZN","NVDA","GOOGL","GOOG","META","BRKB","TSLA","UNH","LLY","JPM","XOM","JNJ","V","PG","AVGO","MA","HD","CVX","MRK","ABBV","PEP","COST","ADBE","KO","CSCO","WMT","TMO","MCD","PFE","CRM","BAC","ACN","CMCSA","LIN","NFLX","ABT","ORCL","DHR","AMD","WFC","DIS","TXN","PM","VZ","INTU","COP","CAT","AMGN","NEE","INTC","UNP","LOW","IBM","BMY","SPGI","RTX","HON","BA","UPS","GE","QCOM","AMAT","NKE","PLD","NOW","BKNG","SBUX","MS","ELV","MDT","GS","DE","ADP","LMT","TJX","T","BLK","ISRG","MDLZ","GILD","MMC","AXP","SYK","REGN","VRTX","ETN","LRCX","ADI","SCHW","CVS","ZTS","CI","CB","AMT","SLB","C","BDX","MO","PGR","TMUS","FI","SO","EOG","BSX","CME","EQIX","MU","DUK","PANW","PYPL","AON","SNPS","ITW","KLAC","HUBB","ICE","APD","SHW","CDNS","CSX","NOC","CL","MPC","HUM","FDX","WM","MCK","TGT","ORLY","HCA","FCX","EMR","PXD","MMM","MCO","ROP","CMG","PSX","MAR","PH","APH","GD","USB","NXPI","AJG","NSC","PNC","VLO","F","MSI","GM","TT","EW","CARR","AZO","ADSK","TDG","ANET","SRE","ECL","OXY","PCAR","ADM","MNST","KMB","PSA","CCI","CHTR","MCHP","MSCI","CTAS","WMB","AIG","STZ","HES","NUE","ROST","AFL","KVUE","AEP","IDXX","D","TEL","JCI","MET","GIS","IQV","EXC","WELL","DXCM","HLT","ON","COF","PAYX","TFC","BIIB","O","FTNT","DOW","TRV","DLR","MRNA","CPRT","ODFL","DHI","YUM","SPG","CTSH","AME","BKR","SYY","A","CTVA","CNC","EL","AMP","CEG","HAL","OTIS","ROK","PRU","DD","KMI","VRSK","LHX","DG","FIS","CMI","CSGP","FAST","PPG","GPN","GWW","HSY","BK","XEL","DVN","EA","NEM","ED","URI","VICI","PEG","KR","RSG","LEN","PWR","WST","COR","OKE","VMC","KDP","WBD","ACGL","ALL","IR","CDW","FANG","MLM","PCG","DAL","EXR","FTV","AWK","IT","KHC","GEHC","WEC","HPQ","EIX","CBRE","APTV","ANSS","MTD","DLTR","AVB","GDDY","ALGN","LYB","TROW","GLW","EFX","WY","ZBH","XYL","SBAC","RMD","TSCO","EBAY","KEYS","CHD","STT","DFS","HIG","ALB","STE","ES","TTWO","MPWR","CAH","EQR","RCL","WTW","HPE","DTE","GPC","BR","ULTA","FICO","CTRA","BAX","AEE","MTB","MKC","ETR","WAB","DOV","FE","RJF","INVH","FLT","CLX","TDY","TRGP","DRI","LH","HOLX","VRSN","MOH","LUV","PPL","ARE","NVR","COO","WBA","PHM","NDAQ","HWM","RF","CNP","IRM","LVS","FITB","EXPD","VTR","FSLR","PFG","BRO","WDAY","IEX","BG","ATO","FDS","ENPH","MAA","CMS","IFF","BALL","SWKS","CINF","NTAP","STLD","UAL","WAT","OMC","TER","CCL","JBHT","TPL","TYL","HBAN","K","GRMN","CBOE","NTRS","TSN","AKAM","EG","ESS","EQT","TXT","EXPE","SJM","PTC","DGX","AVY","RVTY","BBY","CF","CAG","EPAM","AMCR","LW","PAYC","SNA","AXON","POOL","SYF","SWK","ZBRA","DPZ","PKG","CFG","LDOS","VTRS","PODD","LKQ","MOS","APA","EVRG","TRMB","MGM","NDSN","WDC","MAS","LNT","IPG","MTCH","STX","KMX","TECH","WRB","BFB","LYV","IP","UDR","AES","CE","INCY","L","TAP","GEN","CPT","KIM","JKHY","HRL","HST","FMC","CZR","PEAK","CDAY","PNR","NI","CHRW","HSIC","CRL","REG","APO","TFX","KEY","GL","EMN","WYNN","ALLE","PLTR","FFIV","BWA","BXP","MKTX","ROL","JNPR","PNW","DELL","BLDR","FOXA","AOS","HAS","HII","NRG","CPB","UHS","ERIE","WRK","KKR","LII","GEV","BBWI","NWSA","TPR","PARA","SMCI","BEN","AIZ","NCLH","GNRC","FRT","IVZ","SOLV","CRWD","DVA","JBL","LULU","DECK","UBER","MHK","RL","VLTO","FOX","BX","ABNB","NWS"],
        "start_date": "2000-01-01",
        "end_date": None,      # None = up to latest
        "benchmark": "SPY"
    },
    "labels": {                
        "horizon_days": 126,
        "excess_over_risk_free": True,
        "return_threshold": 0.00,
        "dd_window_days": 126,
        "max_drawdown_limit": -0.30,
        "method": "forward_return"
    },
    "POLYGON_API_KEY": "1QI50hs4oNKNjK0pmMmz2UmbLATzpmVA",
    "DATA": "./data",
    "ART": "./artifacts"
}

with open("./config/config.yml","w") as f:
    yaml.safe_dump(cfg, f)
print("Created folders and config")