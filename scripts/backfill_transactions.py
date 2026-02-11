import os
import pandas as pd
import numpy as np
from scripts.data_loading import consumers, accounts, transactions


def build_backfill_df():    
    accounts_clean = accounts.dropna(subset = ['prism_consumer_id', 'balance_date', 'balance'])
    transactions_clean = transactions.dropna(subset = ['prism_consumer_id', 'amount', 'credit_or_debit', 'posted_date'])
    
    df = pd.DataFrame(columns=["prism_consumer_id", "balance", "date", "credit_or_debit", "amount_change"])
        
    # ---- starting balance per consumer (checking) ----
    acc_checking = accounts_clean.loc[
        accounts_clean["account_type"].eq("CHECKING"),
        ["prism_consumer_id", "balance", "balance_date"]
    ].sort_values(["prism_consumer_id", "balance_date"])
    
    first_checking = (
        acc_checking.drop_duplicates("prism_consumer_id", keep="first")
        .set_index("prism_consumer_id")
    )
    
    # ---- transactions grouped + SORTED within consumer ----
    tx_groups = {
        cid: g.sort_values("posted_date")[["prism_consumer_id", "posted_date", "credit_or_debit", "amount"]]
              .to_records(index=False)
        for cid, g in transactions_clean.groupby("prism_consumer_id", sort=False)
    }
    
    # ---- build running balance in chronological order ----
    rows = []
    for consumer_id in accounts_clean["prism_consumer_id"].unique():
        if consumer_id not in first_checking.index:
            continue
    
        start_balance = float(first_checking.at[consumer_id, "balance"])
        start_dt = first_checking.at[consumer_id, "balance_date"]
    
        running = start_balance
    
        rows.append({
            "prism_consumer_id": consumer_id,
            "date": start_dt,
            "balance": running,
            "credit_or_debit": "starting value",
            "amount_change": 0.0,
        })
    
        for tx in tx_groups.get(consumer_id, ()):
            if tx.credit_or_debit == "CREDIT":
                running += float(tx.amount)
            elif tx.credit_or_debit == "DEBIT":
                running -= float(tx.amount)
    
            rows.append({
                "prism_consumer_id": consumer_id,
                "date": tx.posted_date,
                "balance": running,
                "credit_or_debit": tx.credit_or_debit,
                "amount_change": float(tx.amount),
            })
    
    rows_df = pd.DataFrame(rows)
    
    # add DQ_TARGET
    rows_df = rows_df.merge(
        consumers[["prism_consumer_id", "DQ_TARGET"]],
        on="prism_consumer_id",
        how="left",
        validate="m:1"
    )

    return rows_df


