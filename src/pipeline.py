import numpy as np
import pandas as pd

def snap2feat(df:pd.DataFrame, levels=10):
    print("The dataset was successfully read. Producing feature engineering...")

    df.drop(
        labels=["exchange", "symbol", "timestamp", "local_timestamp"], axis=1, inplace=True
    )

    asks = df.filter(regex=("asks"))
    asks_price = np.array([asks[column].to_numpy() for column in asks if "price" in column])
    asks_volume = np.array(
        [asks[column].to_numpy() for column in asks if "amount" in column]
    )

    bids = df.filter(regex=("bids"))
    bids_price = np.array([bids[column].to_numpy() for column in bids if "price" in column])
    bids_volume = np.array(
        [bids[column].to_numpy() for column in bids if "amount" in column]
    )

    n = levels
    length = len(df)

    dataset = pd.DataFrame()

    for level in np.arange(start=0, stop=n, step=1):
        # v_1
        dataset[f"p_ask_{level + 1}"] = asks_price[level]
        dataset[f"v_ask_{level + 1}"] = asks_volume[level]
        dataset[f"p_bid_{level + 1}"] = bids_price[level]
        dataset[f"v_bid_{level + 1}"] = bids_volume[level]

    for level in np.arange(start=0, stop=n, step=1):
        # v_2
        dataset[f"p_ask_{level + 1} - p_bid_{level + 1}"] = (
            asks_price[level] - bids_price[level]
        )

    for level in np.arange(start=0, stop=n, step=1):
        # v_3
        dataset[f"p_ask_{n} - p_ask_{1}"] = asks_price[n - 1] - asks_price[0]
        dataset[f"p_bid_{1} - p_bid_{n}"] = bids_price[0] - bids_price[n - 1]
        dataset[f"|p_ask_{level + 2} - p_ask_{level + 1}|"] = np.absolute(
            asks_price[level + 1] - asks_price[level]
        )
        dataset[f"|p_bid_{level + 2} - p_bid_{level + 1}|"] = np.absolute(
            bids_price[level + 1] - bids_price[level]
        )

    # v_4
    dataset[f"1 / n * Σ^{n}_i={1}(p_ask_{1}...{n})"] = np.array(
        object=[
            1 / n * np.sum(a=asks_price[:n, i])
            for i in np.arange(start=0, stop=length, step=1)
        ]
    )
    dataset[f"1 / n * Σ^{n}_i={1}(p_bid_{1}...{n})"] = np.array(
        object=[
            1 / n * np.sum(a=bids_price[:n, i])
            for i in np.arange(start=0, stop=length, step=1)
        ]
    )
    dataset[f"1 / n * Σ^{n}_i={1}(v_ask_{1}...{n})"] = np.array(
        object=[
            1 / n * np.sum(a=asks_volume[:n, i])
            for i in np.arange(start=0, stop=length, step=1)
        ]
    )
    dataset[f"1 / n * Σ^{n}_i={1}(v_bid_{1}...{n})"] = np.array(
        object=[
            1 / n * np.sum(a=bids_volume[:n, i])
            for i in np.arange(start=0, stop=length, step=1)
        ]
    )

    # v_5
    dataset[f"Σ^{n}_i={1}(p_ask_{1}...{n} - p_bid_{1}...{n})"] = np.array(
        object=[
            np.sum(a=asks_price[:n, i] - bids_price[:n, i])
            for i in np.arange(start=0, stop=length, step=1)
        ]
    )
    dataset[f"Σ^{n}_i={1}(v_ask_{1}...{n} - v_bid_{1}...{n})"] = np.array(
        object=[
            np.sum(a=asks_volume[:n, i] - bids_volume[:n, i])
            for i in np.arange(start=0, stop=length, step=1)
        ]
    )

    # Mid-price
    dataset[f"(p_ask_{1} + p_bid_{1}) / 2"] = (asks_price[0] + bids_price[0]) / 2

    return dataset


