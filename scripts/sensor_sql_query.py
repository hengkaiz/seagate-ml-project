def sql_query(min_date, max_date, wc):
    return f'''select 
    di.date_tmst as di_recirc_date_tmst ,
    mst1.date_tmst as mst1_date_tmst ,
    mst2.date_tmst as mst2_date_tmst ,
    cs.date_tmst as cs_osr_date_tmst ,
    di.di_recirc_flow ,
    mst1.mst1_flow ,
    mst2.mst2_flow ,
    cs.cs_osr_flow
from (
    select
        dte ,
        date_tmst ,
        station ,
        di_recirc_flow 
    from gold.wash_kpiv_v1
    where to_timestamp(dte, 'yyyy-mm-dd')  between to_timestamp('{min_date}', 'yyyy-mm-dd') and to_timestamp('{max_date}', 'yyyy-mm-dd')
    and (di_prod_1 like '%C%' or 
        di_prod_2 like '%C%' or 
        di_prod_3 like '%C%' or 
        di_prod_4 like '%C%')
    and station in ('{wc}')
) as di
full outer join (
    select
        dte ,
        date_tmst ,
        station ,
        mst1_flow 
    from gold.wash_kpiv_v1
    where to_timestamp(dte, 'yyyy-mm-dd')  between to_timestamp('{min_date}', 'yyyy-mm-dd') and to_timestamp('{max_date}', 'yyyy-mm-dd')
    and (mst1_prod_1 like '%C%' or 
        mst1_prod_2 like '%C%' or 
        mst1_prod_3 like '%C%' or 
        mst1_prod_4 like '%C%')
    and station in ('{wc}')
) as mst1
on di.date_tmst = mst1.date_tmst
full outer join (
    select
        dte ,
        date_tmst ,
        station ,
        mst2_flow 
    from gold.wash_kpiv_v1
    where to_timestamp(dte, 'yyyy-mm-dd')  between to_timestamp('{min_date}', 'yyyy-mm-dd') and to_timestamp('{max_date}', 'yyyy-mm-dd')
    and (mst2_prod_1 like '%C%' or 
        mst2_prod_2 like '%C%' or 
        mst2_prod_3 like '%C%' or 
        mst2_prod_4 like '%C%')
    and station in ('{wc}')
) as mst2
on di.date_tmst = mst2.date_tmst
full outer join (
    select
        dte ,
        date_tmst ,
        station ,
        cs_osr_flow 
    from gold.wash_kpiv_v1
    where to_timestamp(dte, 'yyyy-mm-dd')  between to_timestamp('{min_date}', 'yyyy-mm-dd') and to_timestamp('{max_date}', 'yyyy-mm-dd')
    and (cs_load_prod_1 like '%C%' or 
        cs_load_prod_2 like '%C%' or 
        cs_load_prod_3 like '%C%' or 
        cs_load_prod_4 like '%C%')
    and station in ('{wc}')
) as cs
on di.date_tmst = cs.date_tmst
order by di_recirc_date_tmst
'''