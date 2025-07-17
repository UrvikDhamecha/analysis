# Aave V2 Wallet Credit Scoring Model

## 1. Introduction & Challenge Overview

This project addresses the challenge of developing a machine learning model to assign a **credit score between 0 and 1000** to individual wallets interacting with the Aave V2 decentralized lending protocol. The core objective is to differentiate between reliable, responsible usage (higher scores) and risky, bot-like, or exploitative behavior (lower scores) based solely on historical transaction data.

The provided dataset consists of raw, transaction-level data from the Aave V2 protocol, encompassing actions like `deposit`, `borrow`, `repay`, `redeemunderlying`, and `liquidationcall`.

## 2. Method Chosen: Feature-Driven Rule-Based Scoring

Given the absence of pre-labeled "credit scores" for wallets in the provided dataset, a direct supervised machine learning approach was not feasible without a ground truth. Therefore, this solution employs a **feature-driven, rule-based scoring system**. This approach allows for transparency, explainability, and extensibility, where domain knowledge of DeFi interactions is directly translated into scoring criteria.

The credit score is calculated based on a comprehensive set of engineered features that capture various aspects of a wallet's interaction with the Aave V2 protocol.

## 3. Complete Architecture and Processing Flow

The system operates in a one-step script, `score_wallets.py`, following this flow:

### 3.1. Data Ingestion
* **Input**: The raw, transaction-level data from the Aave V2 protocol, provided in a CSV format (`wallet_transactions.csv`).

### 3.2. Data Preprocessing
This crucial phase transforms the raw data into a usable format for feature engineering:
* **Timestamp Conversion**: The `timestamp` column (Unix timestamp) is converted into readable datetime objects to facilitate time-based feature calculations.
* **Numeric Coercion**: Key financial columns such as `actionData.amount`, `actionData.assetPriceUSD`, and others are converted to numeric types, with errors coerced to `NaN` for robust handling.
* **Transaction Value Estimation (USD)**: This is a critical step. The raw `actionData.amount` is in the native token's smallest unit (e.g., wei for ETH, satoshis for BTC, etc.). To make amounts comparable across different assets, they are converted to approximate USD values.
    * This conversion utilizes `actionData.assetPriceUSD` and a predefined mapping of common cryptocurrency decimal places (e.g., USDC uses 6 decimals, WETH uses 18, WBTC uses 8).
    * For unrecognized `assetSymbol` values, a default of 18 decimals is assumed. While this might lead to inaccuracies in absolute USD values for unknown tokens, it maintains relative comparisons within the dataset and prevents errors.

### 3.3. Feature Engineering
After preprocessing, the data is grouped by `userWallet` to calculate aggregated features for each unique wallet. These features are designed to capture the "behavioral fingerprint" of a wallet:

* **Activity-Based Features**:
    * `total_transactions`: Total number of transactions performed by the wallet. (Indicates overall engagement)
    * `unique_actions`: Number of distinct action types (e.g., deposit, borrow, repay). (Indicates diverse interaction)
    * `duration_of_activity_days`: The total number of days between the first and last transaction. (Indicates long-term engagement)

* **Value-Based Features (in USD)**:
    * `total_deposit_usd`: Sum of all deposited amounts in USD.
    * `total_borrow_usd`: Sum of all borrowed amounts in USD.
    * `total_repay_usd`: Sum of all repaid amounts in USD.
    * `total_redeem_usd`: Sum of all redeemed (withdrawn) amounts in USD.
    * `net_deposit_usd`: `total_deposit_usd - total_redeem_usd`. (Positive indicates net capital inflow)
    * `net_borrow_usd`: `total_borrow_usd - total_repay_usd`. (Positive indicates outstanding debt)
    * `repay_to_borrow_ratio`: `total_repay_usd / total_borrow_usd`. (Closer to 1 indicates good repayment behavior; 0 if no borrows).
    * `borrow_to_deposit_ratio`: `total_borrow_usd / total_deposit_usd`. (Higher indicates higher leverage/risk; 0 if no deposits).

* **Risk-Related Features**:
    * `num_liquidated_as_user`: Count of `liquidationcall` actions where the wallet was the `userId` (i.e., the wallet got liquidated). (Strong negative indicator of risk).
    * `num_liquidations_initiated`: Count of `liquidationcall` actions where the wallet was the `callerId` or `liquidatorId` (i.e., the wallet initiated a liquidation). (Could indicate active protocol participation, but needs careful interpretation if solely based on this).
    * `has_outstanding_debt`: A binary flag (1 if `net_borrow_usd` > 0, else 0).

### 3.4. Credit Scoring Logic

A rule-based system assigns points based on the engineered features:

* **Base Score**: All wallets start with a base score of 500.
* **Positive Behavior Rewards**:
    * `total_transactions`: Small positive increment for overall activity.
    * `unique_actions`: Reward for diverse engagement with the protocol.
    * `duration_of_activity_days`: Points for long-term participation.
    * `net_deposit_usd`: Reward for significant net capital contributions (capped to prevent extreme bias).
    * `repay_to_borrow_ratio`: Strong reward for high repayment reliability.
* **Negative Behavior Penalties**:
    * `num_liquidated_as_user`: Significant deduction for being liquidated (major risk indicator).
    * `has_outstanding_debt`: Penalty for having unrepaid debt.
    * `repay_to_borrow_ratio`: Specific penalties for very low (e.g., < 0.5) or zero repayment ratios.

* **Normalization**: The raw calculated scores are then scaled using Min-Max normalization to fit precisely within the 0-1000 range, and rounded to the nearest integer. This ensures consistency and interpretability across all scores.

### 3.5. Output
* The script generates a Pandas DataFrame containing `userWallet` and their corresponding `credit_score`.
* This output can be easily saved to a CSV file (e.g., `wallet_scores.csv`) for further analysis or integration.

## 4. Extensibility

This architecture is designed to be extensible:
* **New Features**: Additional features can be engineered from the raw data (e.g., average borrow duration, collateral health ratio changes, specific asset-wise behaviors) and easily integrated into the scoring function.
* **Machine Learning Models**: If a labeled dataset with actual credit scores or "good/bad" wallet classifications becomes available in the future, the feature-engineered DataFrame can directly feed into supervised machine learning models (e.g., Linear Regression, Random Forest Regressor, XGBoost) to learn more complex relationships and potentially achieve higher predictive accuracy.
* **Rule Tuning**: The weights and thresholds used in the rule-based scoring can be easily adjusted to reflect updated domain knowledge or to fine-tune the score distribution.
