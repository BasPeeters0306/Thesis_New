# Thesis
My master thesis


Data can be loaded with all the functions .......

However if one of the steps does not work (which might happen if EOD changed its API), then load the data directly from .......


df_Compustat_raw.csv = raw market data from Compustat
df_Compustat_raw_SPH.csv = raw market data from Compustat but filtered so that only the historical S&P500 index is included
df_Compustat_raw_SPH_variables.csv = same as df_Compustat_raw_SPH, but now additional variables are added
df_RP_raw.csv = raw RavenPack data, every row contains information about one article
df_SESI.csv = SESI scores created from df_RP_raw.csv
df.csv = df_Compustat_raw_SPH_variables but then the SESI score is added for every row and a 0 is added where there is no SESI score for that stock and day, This is the dataframe which we will work with. Outliers are clipped already in this dataframe.s Load this data from the markdown: Load prepared data