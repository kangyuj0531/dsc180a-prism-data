"""
Feature creation for PRISM financial data
"""

import pandas as pd
import numpy as np


# ============================================================================
# Helper Functions
# ============================================================================

def _safe_div(a, b):
    return a / (b + 1e-9)


def _pct(x, thresh):
    return float((x < thresh).mean()) if len(x) else np.nan


def _trend(y):
    """Slope of y over time index (simple linear fit)."""
    y = np.asarray(y, dtype=float)
    if len(y) < 2:
        return np.nan
    x = np.arange(len(y), dtype=float)
    return float(np.polyfit(x, y, 1)[0])


def _last_window(df, days):
    if df.empty:
        return df
    cutoff = df["date"].max() - pd.Timedelta(days=days)
    return df[df["date"] >= cutoff]


def _window_stats(g, col, days, prefix):
    d = _last_window(g, days)
    s = d[col]
    out = {
        f"{prefix}__mean__{days}d": float(s.mean()) if len(s) else np.nan,
        f"{prefix}__median__{days}d": float(s.median()) if len(s) else np.nan,
        f"{prefix}__min__{days}d": float(s.min()) if len(s) else np.nan,
        f"{prefix}__max__{days}d": float(s.max()) if len(s) else np.nan,
        f"{prefix}__std__{days}d": float(s.std()) if len(s) else np.nan,
        f"{prefix}__trend__{days}d": _trend(s.values),
    }
    return out


def _window_counts(g, days, prefix):
    d = _last_window(g, days)
    out = {
        f"{prefix}__n_days__{days}d": int(d["date"].nunique()) if len(d) else 0,
        f"{prefix}__n_tx__{days}d": int(d["n_tx"].sum()) if len(d) else 0,
    }
    return out


# ============================================================================
# Feature Creation Functions
# ============================================================================

def prepare_daily_data(df):
    """
    Convert backfill dataframe to end-of-day series.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Backfill dataframe with transaction-level data
        
    Returns:
    --------
    pd.DataFrame
        Daily aggregated data with one row per consumer per day
    """
    print("Preparing daily data...")
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(["prism_consumer_id", "date"])
    df["day"] = df["date"].dt.normalize()

    rows_daily = (
        df.groupby(["prism_consumer_id", "day"], as_index=False)
               .agg(
                   balance=("balance", "last"),              # end-of-day balance
                   DQ_TARGET=("DQ_TARGET", "max"),           # label
                   amount_change=("amount_change", "sum"),   # net daily change (optional)
                   n_tx=("amount_change", "size")            # number of intraday rows (optional)
               )
               .rename(columns={"day": "date"})
    )

    rows_daily["date"] = pd.to_datetime(rows_daily["date"])
    rows_daily = rows_daily.sort_values(["prism_consumer_id", "date"])
    
    print(f"Daily data shape: {rows_daily.shape}")
    return rows_daily


def create_balance_features(rows_daily):
    """
    Create balance-based features from daily data.
    
    Parameters:
    -----------
    rows_daily : pd.DataFrame
        Daily aggregated data
        
    Returns:
    --------
    pd.DataFrame
        Balance features aggregated by consumer
    """
    print("Creating balance features...")
    bal_all = rows_daily.groupby("prism_consumer_id").agg(
        balance__mean__all=("balance", "mean"),
        balance__median__all=("balance", "median"),
        balance__min__all=("balance", "min"),
        balance__max__all=("balance", "max"),
        balance__std__all=("balance", "std"),
        balance__pct_negative__all=("balance", lambda x: (x < 0).mean()),
        balance__pct_below_100__all=("balance", lambda x: (x < 100).mean()),
        balance__pct_below_500__all=("balance", lambda x: (x < 500).mean()),
        n_days__all=("date", "nunique"),
        # Total account history length: span from first to last observed date (in days)
        account__history_days__all=("date", lambda x: (x.max() - x.min()).days),
    )
    print(f"Balance features shape: {bal_all.shape}")
    return bal_all


def window_daily_features(df, days):
    """
    Create daily features for a specific time window.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Daily aggregated data
    days : int
        Number of days to look back
        
    Returns:
    --------
    pd.DataFrame
        Features for the specified window
    """
    max_date = df.groupby("prism_consumer_id")["date"].max()
    tmp = df.join(max_date.rename("max_date"), on="prism_consumer_id")
    tmp = tmp[tmp["date"] >= (tmp["max_date"] - pd.Timedelta(days=days))]

    out = tmp.groupby("prism_consumer_id").agg(
        **{
            f"balance__mean__{days}d": ("balance", "mean"),
            f"balance__min__{days}d": ("balance", "min"),
            f"balance__std__{days}d": ("balance", "std"),
            f"balance__pct_negative__{days}d": ("balance", lambda x: (x < 0).mean()),
            f"cashflow__net__{days}d": ("amount_change", "sum"),
            f"cashflow__mean_daily__{days}d": ("amount_change", "mean"),
            f"cashflow__volatility__{days}d": ("amount_change", "std"),
            f"n_tx__{days}d": ("n_tx", "sum"),
            f"n_days__{days}d": ("date", "nunique"),
        }
    )
    return out


def create_daily_window_features(rows_daily):
    """
    Create daily features for multiple time windows.
    
    Parameters:
    -----------
    rows_daily : pd.DataFrame
        Daily aggregated data
        
    Returns:
    --------
    tuple of pd.DataFrame
        Features for 30, 60, 90, and 180 day windows
    """
    print("Creating daily window features...")
    daily_30  = window_daily_features(rows_daily, 30)
    daily_60  = window_daily_features(rows_daily, 60)
    daily_90  = window_daily_features(rows_daily, 90)
    daily_180 = window_daily_features(rows_daily, 180)
    
    print(f"  30d: {daily_30.shape}")
    print(f"  60d: {daily_60.shape}")
    print(f"  90d: {daily_90.shape}")
    print(f"  180d: {daily_180.shape}")
    
    return daily_30, daily_60, daily_90, daily_180


def create_transaction_features(df):
    """
    Create transaction-based features.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Backfill dataframe with transaction-level data
        
    Returns:
    --------
    pd.DataFrame
        Transaction features aggregated by consumer
    """
    print("Creating transaction features...")
    tx = df.copy()
    tx["date"] = pd.to_datetime(tx["date"])

    tx["is_credit"] = tx["credit_or_debit"].eq("CREDIT")
    tx["is_debit"]  = tx["credit_or_debit"].eq("DEBIT")

    # totals + counts
    tx_all = tx.groupby("prism_consumer_id").agg(
        tx__n__all=("amount_change", "size"),
        tx__std_amount__all=("amount_change", "std"),
        credit__total__all=("amount_change", lambda x: x[x > 0].sum()),
        debit__total__all=("amount_change", lambda x: np.abs(x[x < 0]).sum()),
        tx__max_credit__all=("amount_change", lambda x: x[x > 0].max() if (x > 0).any() else 0.0),
        tx__max_debit__all=("amount_change", lambda x: np.abs(x[x < 0]).max() if (x < 0).any() else 0.0),
    )

    tx_all["credit_debit__ratio__all"] = tx_all["credit__total__all"] / (tx_all["debit__total__all"] + 1e-9)
    
    print(f"Transaction features shape: {tx_all.shape}")
    return tx_all


def create_category_features(transactions, category_mapping, topk=30):
    """
    Create category-based features from transaction data.
    
    Parameters:
    -----------
    transactions : pd.DataFrame
        Transaction data
    category_mapping : pd.DataFrame
        Category mapping data
    topk : int
        Number of top categories to include
        
    Returns:
    --------
    tuple of pd.DataFrame
        All-time and 90-day category features
    """
    print(f"Creating category features (top {topk} categories)...")
    
    cat_map = category_mapping.rename(
        columns={"category_id": "category", "category": "category_name"}
    )

    txc = transactions.merge(cat_map, on="category", how="left").copy()
    txc["posted_date"] = pd.to_datetime(txc["posted_date"])
    txc["signed_amount"] = np.where(txc["credit_or_debit"].eq("CREDIT"),
                                    txc["amount"].astype("float32"),
                                    -txc["amount"].astype("float32"))

    top_cats = txc["category"].value_counts().head(topk).index
    txc = txc[txc["category"].isin(top_cats)]

    # all-time
    cat_all = (
        txc.groupby(["prism_consumer_id", "category"])
           .agg(cat_net_total=("signed_amount", "sum"),
                cat_n=("signed_amount", "count"))
           .unstack(fill_value=0)
    )
    cat_all.columns = [f"cat_{int(c)}__{stat}__all" for stat, c in cat_all.columns]

    # 90d window
    max_date = txc.groupby("prism_consumer_id")["posted_date"].max()
    tmp = txc.join(max_date.rename("max_date"), on="prism_consumer_id")
    tmp = tmp[tmp["posted_date"] >= (tmp["max_date"] - pd.Timedelta(days=90))]

    cat_90 = (
        tmp.groupby(["prism_consumer_id", "category"])
           .agg(cat_net_total=("signed_amount", "sum"),
                cat_n=("signed_amount", "count"))
           .unstack(fill_value=0)
    )
    cat_90.columns = [f"cat_{int(c)}__{stat}__90d" for stat, c in cat_90.columns]
    
    print(f"Category features (all-time): {cat_all.shape}")
    print(f"Category features (90d): {cat_90.shape}")
    
    return cat_all, cat_90, txc


def create_group_category_features(txc):
    """
    Create grouped category features (income, essentials, discretionary).
    
    Parameters:
    -----------
    txc : pd.DataFrame
        Transaction data with categories and signed amounts
        
    Returns:
    --------
    pd.DataFrame
        Grouped category features
    """
    print("Creating grouped category features...")
    
    INCOME_CATS = {
        2,   # DEPOSIT
        3,   # PAYCHECK
        6,   # REFUND
        7,   # INVESTMENT_INCOME
        8,   # OTHER_BENEFITS
        9,   # UNEMPLOYMENT_BENEFITS
        42,  # GOVERNMENT_SERVICES
        45,  # INVESTMENT
        49,  # PENSION
    }

    ESSENTIAL_CATS = {
        11,  # TAX
        12,  # LOAN
        13,  # INSURANCE
        17,  # AUTOMOTIVE
        18,  # GROCERIES
        22,  # ESSENTIAL_SERVICES
        23,  # ACCOUNT_FEES
        26,  # CREDIT_CARD_PAYMENT
        27,  # HEALTHCARE_MEDICAL
        29,  # EDUCATION
        31,  # BILLS_UTILITIES
        32,  # MORTGAGE
        33,  # CHILD_DEPENDENTS
        34,  # RENT
        36,  # AUTO_LOAN
        38,  # DEBT
    }

    DISCRETIONARY_CATS = {
        14,  # FOOD_AND_BEVERAGES
        16,  # GENERAL_MERCHANDISE
        19,  # ATM_CASH
        20,  # ENTERTAINMENT
        21,  # TRAVEL
        24,  # HOME_IMPROVEMENT
        28,  # PETS
        30,  # GIFTS_DONATIONS
        35,  # BNPL
        39,  # FITNESS
        40,  # TRANSPORATION
        46,  # GAMBLING
        48,  # TIME_OR_STUFF
    }

    income_total = (
        txc[txc["category"].isin(INCOME_CATS)]
        .groupby("prism_consumer_id")["signed_amount"]
        .apply(lambda x: x[x > 0].sum())
        .rename("income__total__all")
    )

    ess_total = (
        txc[txc["category"].isin(ESSENTIAL_CATS)]
        .groupby("prism_consumer_id")["signed_amount"]
        .apply(lambda x: np.abs(x[x < 0]).sum())
        .rename("essentials_spend__total__all")
    )

    disc_total = (
        txc[txc["category"].isin(DISCRETIONARY_CATS)]
        .groupby("prism_consumer_id")["signed_amount"]
        .apply(lambda x: np.abs(x[x < 0]).sum())
        .rename("discretionary_spend__total__all")
    )

    group_feats = pd.concat([income_total, ess_total, disc_total], axis=1).fillna(0)
    group_feats["essentials__pct_of_income__all"] = group_feats["essentials_spend__total__all"] / (group_feats["income__total__all"] + 1e-9)
    group_feats["discretionary__pct_of_income__all"] = group_feats["discretionary_spend__total__all"] / (group_feats["income__total__all"] + 1e-9)
    
    print(f"Grouped category features shape: {group_feats.shape}")
    return group_feats


def create_overdraft_fee_features(transactions, category_mapping):
    """
    Create overdraft and fee-related features.
    
    Parameters:
    -----------
    transactions : pd.DataFrame
        Transaction data
    category_mapping : pd.DataFrame
        Category mapping data
        
    Returns:
    --------
    pd.DataFrame
        Overdraft and fee features
    """
    print("Creating overdraft & fee features...")
    
    # Merge with category mapping
    cat_map = category_mapping.rename(
        columns={"category_id": "category", "category": "category_name"}
    )
    txc = transactions.merge(cat_map, on="category", how="left").copy()
    txc["posted_date"] = pd.to_datetime(txc["posted_date"])
    txc["signed_amount"] = np.where(txc["credit_or_debit"].eq("CREDIT"),
                                    txc["amount"].astype("float32"),
                                    -txc["amount"].astype("float32"))
    
    # Identify fee categories
    OVERDRAFT_CAT = 25
    ACCOUNT_FEES_CAT = 23
    
    # All-time features
    overdraft_all = (
        txc[txc["category"] == OVERDRAFT_CAT]
        .groupby("prism_consumer_id")
        .agg(
            overdraft_fee__count__all=("signed_amount", "count"),
            overdraft_fee__total__all=("signed_amount", lambda x: np.abs(x.sum())),
        )
    )
    
    account_fees_all = (
        txc[txc["category"] == ACCOUNT_FEES_CAT]
        .groupby("prism_consumer_id")
        .agg(
            account_fees__count__all=("signed_amount", "count"),
            account_fees__total__all=("signed_amount", lambda x: np.abs(x.sum())),
        )
    )
    
    # 30-day window
    max_date = txc.groupby("prism_consumer_id")["posted_date"].max()
    txc_30 = txc.join(max_date.rename("max_date"), on="prism_consumer_id")
    txc_30 = txc_30[txc_30["posted_date"] >= (txc_30["max_date"] - pd.Timedelta(days=30))]
    
    overdraft_30 = (
        txc_30[txc_30["category"] == OVERDRAFT_CAT]
        .groupby("prism_consumer_id")
        .agg(
            overdraft_fee__count__30d=("signed_amount", "count"),
            overdraft_fee__total__30d=("signed_amount", lambda x: np.abs(x.sum())),
        )
    )
    
    account_fees_30 = (
        txc_30[txc_30["category"] == ACCOUNT_FEES_CAT]
        .groupby("prism_consumer_id")
        .agg(
            account_fees__count__30d=("signed_amount", "count"),
            account_fees__total__30d=("signed_amount", lambda x: np.abs(x.sum())),
        )
    )
    
    # 90-day window
    txc_90 = txc.join(max_date.rename("max_date"), on="prism_consumer_id")
    txc_90 = txc_90[txc_90["posted_date"] >= (txc_90["max_date"] - pd.Timedelta(days=90))]
    
    overdraft_90 = (
        txc_90[txc_90["category"] == OVERDRAFT_CAT]
        .groupby("prism_consumer_id")
        .agg(
            overdraft_fee__count__90d=("signed_amount", "count"),
            overdraft_fee__total__90d=("signed_amount", lambda x: np.abs(x.sum())),
        )
    )
    
    account_fees_90 = (
        txc_90[txc_90["category"] == ACCOUNT_FEES_CAT]
        .groupby("prism_consumer_id")
        .agg(
            account_fees__count__90d=("signed_amount", "count"),
            account_fees__total__90d=("signed_amount", lambda x: np.abs(x.sum())),
        )
    )
    
    # Combine all fee features
    fee_feats = pd.concat([
        overdraft_all, account_fees_all,
        overdraft_30, account_fees_30,
        overdraft_90, account_fees_90
    ], axis=1).fillna(0)
    
    # Total fees
    fee_feats["total_fees__all"] = fee_feats["overdraft_fee__total__all"] + fee_feats["account_fees__total__all"]
    fee_feats["total_fees__30d"] = fee_feats["overdraft_fee__total__30d"] + fee_feats["account_fees__total__30d"]
    fee_feats["total_fees__90d"] = fee_feats["overdraft_fee__total__90d"] + fee_feats["account_fees__total__90d"]
    
    print(f"Overdraft & fee features shape: {fee_feats.shape}")
    return fee_feats


def create_low_balance_risk_features(rows_daily):
    """
    Create low balance risk indicators.
    
    Parameters:
    -----------
    rows_daily : pd.DataFrame
        Daily aggregated data
        
    Returns:
    --------
    pd.DataFrame
        Low balance risk features
    """
    print("Creating low balance risk features...")
    
    def calc_balance_risk(g, days):
        """Calculate balance risk metrics for a time window."""
        if days is not None:
            cutoff = g["date"].max() - pd.Timedelta(days=days)
            g = g[g["date"] >= cutoff]
        
        bal = g["balance"]
        
        # Count days in different risk zones
        days_below_zero = (bal < 0).sum()
        days_below_50 = (bal < 50).sum()
        days_below_100 = (bal < 100).sum()
        
        # Consecutive negative days
        is_negative = (bal < 0).astype(int)
        consecutive_neg = 0
        if is_negative.any():
            changes = is_negative.diff().ne(0).cumsum()
            consecutive_neg = is_negative.groupby(changes).sum().max()
        
        # Zero crossings (going from positive to negative)
        zero_crossings = ((bal.shift(1) >= 0) & (bal < 0)).sum()
        
        return pd.Series({
            "days_below_zero": days_below_zero,
            "days_below_50": days_below_50,
            "days_below_100": days_below_100,
            "consecutive_negative_days_max": consecutive_neg,
            "zero_crossings_count": zero_crossings,
        })
    
    # All-time
    risk_all = rows_daily.groupby("prism_consumer_id").apply(
        lambda g: calc_balance_risk(g, None)
    )
    risk_all.columns = [f"balance__{c}__all" for c in risk_all.columns]
    
    # 30-day window
    risk_30 = rows_daily.groupby("prism_consumer_id").apply(
        lambda g: calc_balance_risk(g, 30)
    )
    risk_30.columns = [f"balance__{c}__30d" for c in risk_30.columns]
    
    # 90-day window
    risk_90 = rows_daily.groupby("prism_consumer_id").apply(
        lambda g: calc_balance_risk(g, 90)
    )
    risk_90.columns = [f"balance__{c}__90d" for c in risk_90.columns]
    
    risk_feats = pd.concat([risk_all, risk_30, risk_90], axis=1).fillna(0)
    
    print(f"Low balance risk features shape: {risk_feats.shape}")
    return risk_feats


def create_income_regularity_features(transactions, category_mapping):
    """
    Create income regularity and stability features.
    
    Parameters:
    -----------
    transactions : pd.DataFrame
        Transaction data
    category_mapping : pd.DataFrame
        Category mapping data
        
    Returns:
    --------
    pd.DataFrame
        Income regularity features
    """
    print("Creating income regularity features...")
    
    # Merge with category mapping
    cat_map = category_mapping.rename(
        columns={"category_id": "category", "category": "category_name"}
    )
    txc = transactions.merge(cat_map, on="category", how="left").copy()
    txc["posted_date"] = pd.to_datetime(txc["posted_date"])
    
    # Income categories
    PAYCHECK_CAT = 3
    INCOME_CATS = {2, 3, 6, 7, 8, 9, 42, 45, 49}  # All income-related categories
    
    # Filter to income transactions
    income_tx = txc[(txc["category"].isin(INCOME_CATS)) & (txc["credit_or_debit"] == "CREDIT")].copy()
    income_tx = income_tx.sort_values(["prism_consumer_id", "posted_date"])
    
    # Paycheck-specific
    paycheck_tx = txc[(txc["category"] == PAYCHECK_CAT) & (txc["credit_or_debit"] == "CREDIT")].copy()
    paycheck_tx = paycheck_tx.sort_values(["prism_consumer_id", "posted_date"])
    
    def calc_income_regularity(g):
        """Calculate income regularity metrics."""
        if len(g) < 2:
            return pd.Series({
                "income__frequency__90d": np.nan,
                "income__coefficient_of_variation": np.nan,
                "income__avg_days_between__90d": np.nan,
                "income__count__90d": len(g),
            })
        
        # Filter to last 90 days
        cutoff = g["posted_date"].max() - pd.Timedelta(days=90)
        g_90 = g[g["posted_date"] >= cutoff]
        
        if len(g_90) < 2:
            return pd.Series({
                "income__frequency__90d": np.nan,
                "income__coefficient_of_variation": np.nan,
                "income__avg_days_between__90d": np.nan,
                "income__count__90d": len(g_90),
            })
        
        # Days between income deposits
        days_between = g_90["posted_date"].diff().dt.days.dropna()
        avg_days_between = days_between.mean() if len(days_between) > 0 else np.nan
        
        # Coefficient of variation of income amounts
        amounts = g_90["amount"].astype(float)
        cv = amounts.std() / (amounts.mean() + 1e-9) if len(amounts) > 1 else np.nan
        
        return pd.Series({
            "income__frequency__90d": len(g_90) / 90.0,  # Income events per day
            "income__coefficient_of_variation": cv,
            "income__avg_days_between__90d": avg_days_between,
            "income__count__90d": len(g_90),
        })
    
    def calc_paycheck_regularity(g):
        """Calculate paycheck-specific regularity."""
        if len(g) < 2:
            return pd.Series({
                "paycheck__has_regular": 0,
                "paycheck__consistency_score": np.nan,
                "paycheck__count__90d": len(g),
            })
        
        # Filter to last 90 days
        cutoff = g["posted_date"].max() - pd.Timedelta(days=90)
        g_90 = g[g["posted_date"] >= cutoff]
        
        if len(g_90) < 2:
            return pd.Series({
                "paycheck__has_regular": 0,
                "paycheck__consistency_score": np.nan,
                "paycheck__count__90d": len(g_90),
            })
        
        # Check for regularity (2-4 paychecks per month suggests regular employment)
        has_regular = 1 if 2 <= len(g_90) / 3 <= 4 else 0
        
        # Consistency: inverse of coefficient of variation
        amounts = g_90["amount"].astype(float)
        cv = amounts.std() / (amounts.mean() + 1e-9) if len(amounts) > 1 else 1.0
        consistency = 1.0 / (1.0 + cv)  # Higher is more consistent
        
        return pd.Series({
            "paycheck__has_regular": has_regular,
            "paycheck__consistency_score": consistency,
            "paycheck__count__90d": len(g_90),
        })
    
    income_reg = income_tx.groupby("prism_consumer_id").apply(calc_income_regularity)
    paycheck_reg = paycheck_tx.groupby("prism_consumer_id").apply(calc_paycheck_regularity)
    
    regularity_feats = pd.concat([income_reg, paycheck_reg], axis=1).fillna(0)
    
    print(f"Income regularity features shape: {regularity_feats.shape}")
    return regularity_feats


def create_paycheck_to_paycheck_features(rows_daily, transactions, category_mapping):
    """
    Create paycheck-to-paycheck behavior indicators.
    
    Parameters:
    -----------
    rows_daily : pd.DataFrame
        Daily aggregated data
    transactions : pd.DataFrame
        Transaction data
    category_mapping : pd.DataFrame
        Category mapping data
        
    Returns:
    --------
    pd.DataFrame
        Paycheck-to-paycheck features
    """
    print("Creating paycheck-to-paycheck features...")
    
    # Merge with category mapping
    cat_map = category_mapping.rename(
        columns={"category_id": "category", "category": "category_name"}
    )
    txc = transactions.merge(cat_map, on="category", how="left").copy()
    txc["posted_date"] = pd.to_datetime(txc["posted_date"])
    
    # Income categories
    INCOME_CATS = {2, 3, 6, 7, 8, 9, 42, 45, 49}
    
    # Get income deposit dates
    income_tx = txc[(txc["category"].isin(INCOME_CATS)) & (txc["credit_or_debit"] == "CREDIT")].copy()
    income_tx = income_tx[["prism_consumer_id", "posted_date", "amount"]].sort_values(
        ["prism_consumer_id", "posted_date"]
    )
    
    def calc_p2p_behavior(consumer_id):
        """Calculate paycheck-to-paycheck metrics for a consumer."""
        # Get consumer's daily balances
        daily = rows_daily[rows_daily["prism_consumer_id"] == consumer_id].copy()
        daily = daily.sort_values("date")
        
        # Get consumer's income dates (last 90 days)
        cutoff = daily["date"].max() - pd.Timedelta(days=90)
        daily_90 = daily[daily["date"] >= cutoff]
        income_dates = income_tx[
            (income_tx["prism_consumer_id"] == consumer_id) &
            (income_tx["posted_date"] >= cutoff)
        ]["posted_date"].values
        
        if len(income_dates) < 2 or len(daily_90) < 7:
            return pd.Series({
                "balance__min_before_income__avg__90d": np.nan,
                "balance__depletion_rate__90d": np.nan,
                "days_to_deplete_half_balance__avg__90d": np.nan,
            })
        
        # Find minimum balance before each income deposit
        min_balances_before_income = []
        depletion_days = []
        
        for i, income_date in enumerate(income_dates[1:], start=1):
            prev_income_date = income_dates[i - 1]
            
            # Get balances between income deposits
            between_income = daily_90[
                (daily_90["date"] > prev_income_date) &
                (daily_90["date"] <= income_date)
            ]
            
            if len(between_income) > 0:
                min_bal = between_income["balance"].min()
                min_balances_before_income.append(min_bal)
                
                # Calculate depletion: balance right after income vs. min balance
                start_bal = daily_90[daily_90["date"] >= prev_income_date].iloc[0]["balance"]
                if start_bal > 0:
                    half_bal = start_bal / 2
                    days_to_half = len(between_income[between_income["balance"] <= half_bal])
                    if days_to_half > 0:
                        depletion_days.append(days_to_half)
        
        avg_min_before_income = np.mean(min_balances_before_income) if min_balances_before_income else np.nan
        
        # Depletion rate: how fast balance drops after income
        depletion_rate = np.nan
        if len(daily_90) > 1:
            balance_changes = daily_90["balance"].diff().dropna()
            negative_changes = balance_changes[balance_changes < 0]
            if len(negative_changes) > 0:
                depletion_rate = abs(negative_changes.mean())
        
        avg_days_to_deplete_half = np.mean(depletion_days) if depletion_days else np.nan
        
        return pd.Series({
            "balance__min_before_income__avg__90d": avg_min_before_income,
            "balance__depletion_rate__90d": depletion_rate,
            "days_to_deplete_half_balance__avg__90d": avg_days_to_deplete_half,
        })
    
    # Calculate for all consumers
    consumer_ids = rows_daily["prism_consumer_id"].unique()
    p2p_list = []
    
    for consumer_id in consumer_ids:
        result = calc_p2p_behavior(consumer_id)
        result.name = consumer_id
        p2p_list.append(result)
    
    p2p_feats = pd.DataFrame(p2p_list).fillna(0)
    
    print(f"Paycheck-to-paycheck features shape: {p2p_feats.shape}")
    return p2p_feats


def create_multi_account_features(accounts_df):
    """
    Create features from SAVINGS, CREDIT CARD, and other account types.

    Account interpretation:
      - SAVINGS / MONEYMARKET / CASH MANAGEMENT  → assets (balance = money held)
      - CREDIT CARD                              → liability (balance = debt owed)
      - LINE OF CREDIT / OVERDRAFT               → revolving debt
      - LOAN / MORTGAGE / AUTO / STUDENT /
        HOME EQUITY                              → long-term debt

    Parameters:
    -----------
    accounts_df : pd.DataFrame
        Accounts table (q2-ucsd-acctDF.pqt)

    Returns:
    --------
    pd.DataFrame
        Multi-account features indexed by prism_consumer_id
    """
    print("Creating multi-account features...")

    accts = accounts_df.dropna(subset=["prism_consumer_id", "balance_date", "balance"]).copy()
    accts["balance_date"] = pd.to_datetime(accts["balance_date"])
    accts["acct_type"] = accts["account_type"].str.upper().str.strip()

    # ── per-account latest snapshot ──────────────────────────────
    # If prism_account_id exists, deduplicate per account first so that
    # consumers with many snapshots don't get over-counted.
    id_col = "prism_account_id" if "prism_account_id" in accts.columns else None
    if id_col:
        latest = (
            accts.sort_values("balance_date")
                 .groupby(["prism_consumer_id", id_col], as_index=False)
                 .last()
        )
    else:
        latest = accts.sort_values("balance_date").copy()

    feat_parts = []

    # ── SAVINGS (liquid assets) ───────────────────────────────────
    SAVINGS_TYPES = {"SAVINGS", "MONEYMARKET", "MONEY MARKET", "CASH MANAGEMENT"}
    sav = latest[latest["acct_type"].isin(SAVINGS_TYPES)]
    if len(sav) > 0:
        feat_parts.append(
            sav.groupby("prism_consumer_id")["balance"].agg(
                savings__balance__latest="last",
                savings__balance__mean="mean",
                savings__balance__min="min",
                savings__balance__max="max",
                savings__n_accounts="count",
            )
        )

    # ── CREDIT CARD (short-term revolving debt) ───────────────────
    # Balance = amount owed → higher balance is worse
    cc = latest[latest["acct_type"] == "CREDIT CARD"]
    if len(cc) > 0:
        feat_parts.append(
            cc.groupby("prism_consumer_id")["balance"].agg(
                credit_card__debt__latest="last",
                credit_card__debt__mean="mean",
                credit_card__debt__max="max",   # peak debt
                credit_card__n_accounts="count",
            )
        )

    # ── LINE OF CREDIT + OVERDRAFT (revolving credit lines) ───────
    revolving = latest[latest["acct_type"].isin({"LINE OF CREDIT", "OVERDRAFT"})]
    if len(revolving) > 0:
        feat_parts.append(
            revolving.groupby("prism_consumer_id")["balance"].agg(
                revolving_debt__balance__latest="last",
                revolving_debt__balance__mean="mean",
                revolving_debt__n_accounts="count",
            )
        )

    # ── Long-term debt (LOAN, MORTGAGE, AUTO, STUDENT, HOME EQUITY)
    longterm = latest[latest["acct_type"].isin(
        {"LOAN", "MORTGAGE", "AUTO", "STUDENT", "HOME EQUITY"}
    )]
    if len(longterm) > 0:
        feat_parts.append(
            longterm.groupby("prism_consumer_id")["balance"].agg(
                longterm_debt__balance__total="sum",
                longterm_debt__balance__mean="mean",
                longterm_debt__n_accounts="count",
            )
        )

    if not feat_parts:
        print("  No multi-account features computed (no matching account types found).")
        return pd.DataFrame()

    multi = pd.concat(feat_parts, axis=1).fillna(0)

    # ── Composite features ────────────────────────────────────────
    sav_col  = multi["savings__balance__latest"]      if "savings__balance__latest"      in multi.columns else 0
    cc_col   = multi["credit_card__debt__latest"]     if "credit_card__debt__latest"     in multi.columns else 0
    rev_col  = multi["revolving_debt__balance__latest"] if "revolving_debt__balance__latest" in multi.columns else 0
    lt_col   = multi["longterm_debt__balance__total"]  if "longterm_debt__balance__total"  in multi.columns else 0

    # Net liquid position: liquid savings minus short-term debt obligations
    multi["net_liquid_position__all_accounts"] = sav_col - cc_col - rev_col

    # Total debt across all account types
    multi["total_debt__all_accounts"] = cc_col + rev_col + lt_col

    # Savings-to-debt ratio (higher = cushion against debt)
    multi["savings_to_debt_ratio"] = sav_col / (cc_col + rev_col + lt_col + 1e-9)

    print(f"Multi-account features shape: {multi.shape}")
    return multi


def create_all_features(df, transactions, category_mapping, accounts_df=None):
    """
    Create all features from raw data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Backfill dataframe with transaction-level data
    transactions : pd.DataFrame
        Transaction data
    category_mapping : pd.DataFrame
        Category mapping data
    accounts_df : pd.DataFrame, optional
        Accounts table; when provided, adds SAVINGS / CREDIT CARD /
        LINE OF CREDIT / long-term debt features.
        
    Returns:
    --------
    pd.DataFrame
        Complete feature matrix with DQ_TARGET label
    """
    print("="*70)
    print("FEATURE CREATION PIPELINE")
    print("="*70)
    
    # Prepare daily data
    rows_daily = prepare_daily_data(df)
    
    # Create balance features
    bal_all = create_balance_features(rows_daily)
    
    # Create daily window features
    daily_30, daily_60, daily_90, daily_180 = create_daily_window_features(rows_daily)
    
    # Create transaction features
    tx_all = create_transaction_features(df)
    
    # Create category features
    cat_all, cat_90, txc = create_category_features(transactions, category_mapping)
    
    # Create grouped category features
    group_feats = create_group_category_features(txc)
    
    # Create overdraft & fee features
    fee_feats = create_overdraft_fee_features(transactions, category_mapping)
    
    # Create low balance risk features
    risk_feats = create_low_balance_risk_features(rows_daily)
    
    # Create income regularity features
    income_reg_feats = create_income_regularity_features(transactions, category_mapping)
    
    # Create paycheck-to-paycheck features
    p2p_feats = create_paycheck_to_paycheck_features(rows_daily, transactions, category_mapping)

    # Create multi-account features (SAVINGS, CREDIT CARD, etc.)
    multi_acct_feats = None
    if accounts_df is not None:
        multi_acct_feats = create_multi_account_features(accounts_df)

    # Join all features
    print("\nJoining all features...")
    X = (
        bal_all
        .join([daily_30, daily_60, daily_90, daily_180], how="outer")
        .join(tx_all, how="outer")
        .join(cat_all, how="outer")
        .join(cat_90, how="outer")
        .join(group_feats, how="outer")
        .join(fee_feats, how="outer")
        .join(risk_feats, how="outer")
        .join(income_reg_feats, how="outer")
        .join(p2p_feats, how="outer")
    )
    if multi_acct_feats is not None and len(multi_acct_feats) > 0:
        X = X.join(multi_acct_feats, how="left")
    X = X.replace([np.inf, -np.inf], np.nan)

    y = (
        rows_daily
        .groupby("prism_consumer_id")["DQ_TARGET"]
        .max()
        .rename("DQ_TARGET")
    )

    features_df = X.join(y, how="inner")

    # Exclude consumers without a valid label (DQ_TARGET == NaN)
    features_df = features_df[features_df["DQ_TARGET"].notna()].copy()
    
    print("\n" + "="*70)
    print(f"FINAL FEATURE MATRIX")
    print("="*70)
    print(f"Shape: {features_df.shape}")
    print(f"Number of features: {features_df.shape[1] - 1}")
    print(f"Number of consumers: {features_df.shape[0]}")
    print("="*70)
    
    return features_df


def print_feature_summary(features_df):
    """
    Print summary statistics for all features.
    
    Parameters:
    -----------
    features_df : pd.DataFrame
        Feature matrix with DQ_TARGET label
    """
    X = features_df.drop(columns=["DQ_TARGET"]).select_dtypes(include=[np.number]).copy()

    summary = pd.DataFrame({
        "feature": X.columns,
        "nonzero_rate": (X != 0).mean().values,
        "mean": X.mean().values,
        "std": X.std().values,
    })

    summary = summary.sort_values("nonzero_rate", ascending=False)
    
    print("\nTop 30 features by non-zero rate:")
    print(summary.head(30).to_string())
    
    return summary


def print_feature_groups(features_df):
    """
    Print features organized by logical groups.
    
    Parameters:
    -----------
    features_df : pd.DataFrame
        Feature matrix with DQ_TARGET label
    """
    feature_cols = [c for c in features_df.columns if c != "DQ_TARGET"]

    groups = {
        "Balance behavior": [c for c in feature_cols if c.startswith("balance__") and "before_income" not in c],
        "Cashflow behavior": [c for c in feature_cols if c.startswith("cashflow__")],
        "Transaction activity": [c for c in feature_cols if c.startswith("tx__") or c.startswith("credit_") or c.startswith("debit_")],
        "Category-level behavior": [c for c in feature_cols if c.startswith("cat_")],
        "Income & spending burden": [c for c in feature_cols if c.startswith("income__") or "spend" in c or "pct_of_income" in c],
        "Overdraft & fees": [c for c in feature_cols if "overdraft" in c or "fee" in c],
        "Low balance risk": [c for c in feature_cols if any(x in c for x in ["days_below", "consecutive_negative", "zero_crossings"])],
        "Income regularity": [c for c in feature_cols if c.startswith("paycheck__") or "coefficient_of_variation" in c or "frequency" in c or "days_between" in c],
        "Paycheck-to-paycheck": [c for c in feature_cols if "before_income" in c or "depletion" in c or "deplete" in c],
        "Multi-account (savings/debt)": [c for c in feature_cols if any(x in c for x in [
            "savings__", "credit_card__", "revolving_debt__", "longterm_debt__",
            "net_liquid_position", "total_debt__all_accounts", "savings_to_debt_ratio"
        ])],
    }

    print("\n" + "="*70)
    print("FEATURE GROUPS")
    print("="*70)
    
    for title, cols in groups.items():
        print(f"\n{title} ({len(cols)} features):")
        for c in cols:
            print("  -", c)
    
    print("="*70)
