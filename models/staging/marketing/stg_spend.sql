with spend as (

	select *

	from {{ source('dash_ad_spend', 'campaign_spend') }}
)

select * from spend
