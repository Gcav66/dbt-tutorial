with revenue as (

	select 
	    year,
	    month,
	    revenue

	from {{ source('dash_ad_spend', 'monthly_revenue') }}
)

select * from revenue
