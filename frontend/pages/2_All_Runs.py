import streamlit as st
from utils import all_runs, get_run_details
import pandas as pd
import os

st.set_page_config(
    page_title="All Runs",
    initial_sidebar_state="collapsed",
)

st.title("All Runs")


def convert_to_dataframe(documents):
    """Converts the list of documents into a Pandas DataFrame with specific fields.

    Args:
        documents (list): A list of documents from the MongoDB collection.

    Returns:
        DataFrame: A Pandas DataFrame containing specific fields.
    """
    data = []
    for doc in documents:
        row = {
            "run_id": doc["run_id"],
            "created_at": pd.to_datetime(doc["created_at"], unit="s"),
            "converged": doc["converged"],
            "problem_type": doc["settings"]["problem_type"],
            "objective_function": doc["settings"]["objective_function"],
        }
        data.append(row)

    df = pd.DataFrame(data)
    return df


past_runs = all_runs()
if len(past_runs) == 0:
    st.write("No runs found.")
else:
    runs = convert_to_dataframe(past_runs)

    st.dataframe(
        runs,
        width=None,
        height=None,
        hide_index=True,
    )
    run_ids = runs["run_id"].tolist()

    run_id = st.selectbox("Select a run ID", run_ids)

    if st.button("View Run Details"):
        documents = get_run_details(run_id)
        latest_iteration = documents[-1]
        information = {
            "Run ID":
            latest_iteration["run_id"],
            "Created At":
            pd.to_datetime(latest_iteration["created_at"], unit="s"),
            "Finished At":
            pd.to_datetime(latest_iteration["finished_at"], unit="s"),
            "Global Best Score":
            latest_iteration["global_best"]["score"],
            "Global Best Position":
            latest_iteration["global_best"]["position"],
            "Converged":
            latest_iteration["converged"],
            "Convergence Iteration":
            latest_iteration["convergence_iteration"],
            "Convergence Rate":
            latest_iteration["convergence_rate"],
            "Score based Precision":
            latest_iteration["score_precision"][-1],
            "Position based Precision":
            latest_iteration["position_precision"][-1],
        }

        st.write(information)

        score_precision = latest_iteration["score_precision"]
        position_precision = latest_iteration["position_precision"]

        col1, col2 = st.columns(2)
        with col1:
            st.line_chart(score_precision, use_container_width=True)
        with col2:
            st.line_chart(position_precision, use_container_width=True)

        filepath = os.path.join(f"/pso_media/{run_id}.mp4")
        st.video(filepath)
