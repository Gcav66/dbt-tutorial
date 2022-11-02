with budget as (

	select *

	from {{ source('dash_ad_spend', 'budget_allocations_and_roi') }}
)

select * from budget
