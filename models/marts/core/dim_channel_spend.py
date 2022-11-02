#dim_channel_spend.py

import pandas as pd

def model(dbt, session):
    dbt.config(
        materialized="table",
        packages=["pandas"]
    )

    local_df = dbt.ref("stg_spend").to_pandas()
    #### Need to rewrite to "standard" pandas from Snowpark
    local_df['DT_DATE'] = pd.to_datetime(local_df['DATE'])
    local_df['MONTH'] = local_df['DT_DATE'].dt.month
    local_df['YEAR'] = local_df['DT_DATE'].dt.year
    pre_df = local_df.groupby(['YEAR', 'MONTH', 'CHANNEL']).sum()
    foo_df = pre_df.reset_index()
    results = foo_df.pivot_table(values="TOTAL_COST",index=["YEAR", "MONTH"], columns="CHANNEL")
    rev_df = dbt.ref("stg_revenue").to_pandas()
    rev_df = rev_df.set_index(['YEAR', 'MONTH'])
    spend_rev = results.join(rev_df, ["YEAR","MONTH"])
    spend_rev = spend_rev.dropna()
    return spend_rev
