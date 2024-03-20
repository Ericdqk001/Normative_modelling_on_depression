from pathlib import Path

import pandas as pd


def main():
    """Prepare the CBCL data for LCA analysis by selecting the baseline data and
    removing missing values."""
    data_path = Path(
        "raw_data/core/mental-health/mh_p_cbcl.csv",
    )

    cbcl_t_vars_path = Path(
        "src/behaviour/files/cbcl_8_dim_t.csv",
    )

    cbcl = pd.read_csv(data_path, index_col=0)

    cbcl_t_vars_df = pd.read_csv(cbcl_t_vars_path)

    cbcl_t_vars = cbcl_t_vars_df["var_name"].tolist()

    # Select the baseline data
    baseline_cbcl = cbcl[cbcl["eventname"] == "baseline_year_1_arm_1"]

    # Filter columns with t variables
    filtered_cbcl = baseline_cbcl[cbcl_t_vars]

    # Remove missing values
    filtered_cbcl = filtered_cbcl.dropna()

    # Create dummy variables using a threshold of 65 for the t scores
    filtered_cbcl = (filtered_cbcl >= 65).astype(int)

    # Save the filtered data
    filtered_cbcl_save_path = Path("processed_data")
    filtered_cbcl.to_csv(
        Path(
            filtered_cbcl_save_path,
            "cbcl_t_no_mis_dummy.csv",
        ),
    )


if __name__ == "__main__":
    main()
