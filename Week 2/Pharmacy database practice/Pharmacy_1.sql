use pharmacy; 

-- to show all the items or data in a table
select *  from products;
select * from category;

-- query to update values inside table 
update products set p_category=4 where p_id in (11,12,13,14,15,16);
update products set p_category=5 where p_id=2;
update products set p_category=7 where p_id=4;

-- sort by ascending order 
select * from products order by total_price ;

-- sort by descending order 
select * from products order by total_price DESC;

-- to show number of medecine of each category and with average price of that category
select p_category,count(*)  as medecines ,avg(total_price) from products group by p_category;

-- to calculate price_per_piece from the piece_per_item and price
select p_id,p_name, price_per_item,total_price, total_price/piece_per_item as price_per_piece from products;

-- to calculate the total_amount of avaible quantity of medecine. To check what is the value of quantity present
select p_id,p_name, price_per_item,total_price,available_quantity ,available_quantity * price_per_item as available_quantity_price,
sum(available_quantity * price_per_item) as available_quantity_total_amount from products GROUP BY 
p_id, p_name, price_per_item, total_price, available_quantity;

-- round the price by one decimal
select p_name,round(price_per_item,1) as rounded_price,price_per_item from products;

-- show the names in uppercase
select ucase(p_name) from products;

-- to show the category and manufacturer name of each product using join
select p_name,p_id,category_name,m_name from products p,category c,manufacturer m 
where p.p_id=m.m_id and p.p_category = c.category_id;
