from pathlib import Path
import sqlite3
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW, CLEAN = ROOT / "data/raw", ROOT / "data/clean"
OUT = ROOT / "outputs"
CLEAN.mkdir(parents=True, exist_ok=True)
OUT.mkdir(exist_ok=True)

def run():
    customers = pd.read_csv(RAW / "customers.csv", keep_default_na=False)
    products = pd.read_csv(RAW / "products.csv")
    orders = pd.read_csv(RAW / "orders.csv", keep_default_na=False)

    # Customer normalization.
    customers["city"] = customers["city"].astype(str).str.strip().str.title()
    gender_map = {"male":"male", "m":"male", "female":"female", "f":"female"}
    customers["gender"] = customers["gender"].astype(str).str.strip().str.lower().map(gender_map).fillna("unknown")
    customers["age"] = pd.to_numeric(customers["age"], errors="coerce")
    customers.loc[~customers["age"].between(10, 100), "age"] = np.nan
    customers["age"] = customers["age"].fillna(
        customers.groupby("city")["age"].transform("median")
    )
    customers["signup_date"] = pd.to_datetime(customers["signup_date"])
    customers["age"] = customers["age"].round().astype(int)

    # Order normalization in the required order.
    orders["order_ts"] = pd.to_datetime(orders["order_ts"])
    orders = orders.drop_duplicates()
    orders = orders.sort_values(["order_ts"]).drop_duplicates("order_id", keep="last")
    orders["quantity"] = pd.to_numeric(orders["quantity"], errors="coerce")
    orders["discount_pct"] = pd.to_numeric(orders["discount_pct"], errors="coerce")
    orders["order_amount"] = pd.to_numeric(orders["order_amount"], errors="coerce")
    orders["delivery_minutes"] = pd.to_numeric(orders["delivery_minutes"], errors="coerce")
    orders.loc[orders["delivery_minutes"] < 0, "delivery_minutes"] = np.nan
    orders = orders.merge(products[["product_id","price_inr","category"]], on="product_id", how="left")
    missing_amt = orders["order_amount"].isna()
    orders.loc[missing_amt, "order_amount"] = (
        orders.loc[missing_amt, "price_inr"] *
        orders.loc[missing_amt, "quantity"] *
        (1 - orders.loc[missing_amt, "discount_pct"] / 100)
    ).round(2)
    completed = orders["status"].eq("completed")
    medians = orders.loc[completed].groupby("category")["delivery_minutes"].transform("median")
    orders.loc[completed, "delivery_minutes"] = orders.loc[completed, "delivery_minutes"].fillna(medians)
    orders = orders.drop(columns=["price_inr","category"])

    # Export clean CSVs.
    customers.to_csv(CLEAN / "customers.csv", index=False, date_format="%Y-%m-%d")
    products.to_csv(CLEAN / "products.csv", index=False)
    orders.to_csv(CLEAN / "orders.csv", index=False, date_format="%Y-%m-%d %H:%M:%S")

    db = ROOT / "data/instamart.db"
    db.parent.mkdir(exist_ok=True)
    with sqlite3.connect(db) as con:
        customers.to_sql("customers", con, if_exists="replace", index=False)
        products.to_sql("products", con, if_exists="replace", index=False)
        orders.to_sql("orders", con, if_exists="replace", index=False)

    # SQL analysis queries.
    queries = {
    "Q1": """SELECT customer_id, SUM(order_amount) total FROM orders
             WHERE status='completed' AND order_amount>500
             GROUP BY customer_id HAVING SUM(order_amount)>8000;""",
    "Q2": """SELECT c.city,p.category,SUM(o.order_amount) revenue
             FROM orders o JOIN customers c USING(customer_id)
             JOIN products p USING(product_id) WHERE o.status='completed'
             GROUP BY c.city,p.category ORDER BY c.city,p.category;""",
    "Q3": """SELECT product_id,product_name,price_inr FROM products
             WHERE price_inr>(SELECT AVG(price_inr) FROM products);""",
    "Q4": """SELECT c.customer_id FROM customers c LEFT JOIN orders o
             ON c.customer_id=o.customer_id AND o.status='completed'
             WHERE o.order_id IS NULL;""",
    "Q5": """WITH spend AS (SELECT customer_id,SUM(order_amount) total
             FROM orders WHERE status='completed' GROUP BY customer_id),
             avg_spend AS (SELECT AVG(total) avg_total FROM spend)
             SELECT customer_id,total FROM spend WHERE total>(SELECT avg_total FROM avg_spend);""",
    "Q6": """WITH spend AS (SELECT c.customer_id,c.city,SUM(o.order_amount) total
             FROM customers c JOIN orders o USING(customer_id)
             WHERE o.status='completed' GROUP BY c.customer_id,c.city)
             SELECT *,RANK() OVER(PARTITION BY city ORDER BY total DESC) rnk,
             DENSE_RANK() OVER(PARTITION BY city ORDER BY total DESC) dense_rnk,
             ROW_NUMBER() OVER(PARTITION BY city ORDER BY total DESC) row_num
             FROM spend ORDER BY city,total DESC;""",
    "Q7": """WITH daily AS (SELECT substr(order_ts,1,10) day,SUM(order_amount) revenue
             FROM orders WHERE status='completed' GROUP BY substr(order_ts,1,10))
             SELECT day,revenue,SUM(revenue) OVER(ORDER BY day) cumulative_revenue
             FROM daily ORDER BY day;"""
    }
    (ROOT / "sql/queries.sql").write_text("\n\n".join(f"-- {k}\n{v}" for k,v in queries.items()), encoding="utf-8")
    with sqlite3.connect(db) as con:
        results = {k: pd.read_sql_query(q, con) for k,q in queries.items()}

    # Auditable assertions for core deterministic checkpoints.
    assert len(customers) == 1200
    assert customers["city"].nunique() == 6
    assert customers["gender"].value_counts().to_dict() == {"female":610,"male":560,"unknown":30}
    assert len(orders) == 12309
    assert orders["status"].value_counts().to_dict() == {"completed":11110,"cancelled":1199}
    assert orders.loc[orders.status.eq("completed"),"delivery_minutes"].isna().sum() == 0
    assert orders.loc[orders.status.eq("cancelled"),"delivery_minutes"].isna().sum() == 1199
    assert round(orders.loc[orders.status.eq("completed"),"order_amount"].sum(),2) == 2998668.80
    (OUT / "part_a_validation.txt").write_text("ALL CHECKPOINTS PASSED\n", encoding="utf-8")
    return customers, products, orders, results

if __name__ == "__main__":
    run()
