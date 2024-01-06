import json

import streamlit as st
from utils import animate_pso, get_objective_functions, run_pso
from utils.logger import get_logger

logger = get_logger(__name__)

st.set_page_config(
    page_title="Playground",
    initial_sidebar_state="expanded",
)

fixed_bounds = {
    "rastrigin": "[[-5.12, 5.12], [-5.12, 5.12]]",
}

st.title("Playground")

available_functions = get_objective_functions()
function_index = next(
    (i for i, s in enumerate(available_functions) if "1d" in s), None)

st.sidebar.title("Settings")

st.sidebar.subheader("Results")
result_layout = st.sidebar.columns(2)
with result_layout[0]:
    decimal_places = st.sidebar.number_input("Decimal Places",
                                             min_value=0,
                                             value=3)
with result_layout[1]:
    convergence_type = st.sidebar.selectbox("Convergence Type",
                                            ["Iteration", "Proportion"])

st.sidebar.subheader("Topology")
available_topologies = ["global", "ring"]
topology = st.sidebar.selectbox("Topology", available_topologies, index=0)

st.sidebar.subheader("Objective")
objective_layout = st.sidebar.columns(2)
with objective_layout[0]:
    problem_type = st.selectbox("Problem Type", ["min", "max"])
with objective_layout[1]:
    chosen_function = st.selectbox("Objective Function",
                                   available_functions,
                                   index=function_index)
    dimension = 1 if "1d" in chosen_function else 2 if "2d" in chosen_function else None

st.sidebar.subheader("Size")
size_layout = st.sidebar.columns(2)
with size_layout[0]:
    num_particles = st.number_input("Particles", min_value=2, value=10, step=1)
with size_layout[1]:
    max_iterations = st.number_input("Iterations",
                                     min_value=1,
                                     value=20,
                                     step=1)

st.sidebar.subheader("Weights")
weights_layout = st.sidebar.columns(3)
with weights_layout[0]:
    cognitive_weight = st.number_input("Cognitive", value=0.5)
with weights_layout[1]:
    social_weight = st.number_input("Social", value=2.0)
with weights_layout[2]:
    inertia_weight = st.number_input("Inertia", value=0.4)

st.sidebar.subheader("Bounds")
if dimension is not None:
    bounds = []
    for i in range(dimension):
        st.sidebar.write(f"Dimension {i+1}")
        bound_layout = st.sidebar.columns(2)
        with bound_layout[0]:
            min_bound = st.number_input("Minimum", value=-5, key=f"min_{i}")
        with bound_layout[1]:
            max_bound = st.number_input("Maximum", value=5, key=f"max_{i}")
        if min_bound >= max_bound:
            st.sidebar.error(
                f"Minimum bound must be less than maximum for Dimension {i+1}")
        bounds.append([min_bound, max_bound])
else:
    bounds = None
    bounds_help = "e.g., [[-10, 10], [-5, 5]])"
    if chosen_function in fixed_bounds:
        bounds_input = fixed_bounds[chosen_function]
        st.sidebar.write(bounds_input)
    else:
        bounds_input = st.sidebar.text_input(
            "Enter bounds as a JSON array",
            help=bounds_help,
            placeholder="[[-10, 10], [-5, 5]]")
    if bounds_input:
        invalid_msg = "Error: Ensure each bound is a pair of [min, max] " + \
                        "where min < max in a JSON array"
        try:
            bounds = json.loads(bounds_input)
            if len(bounds) == 0:
                st.sidebar.error("Error: Empty JSON array")
            for bound in bounds:
                if len(bound) != 2 or bound[0] >= bound[1]:
                    st.sidebar.error(invalid_msg)
        except json.JSONDecodeError:
            st.sidebar.error(invalid_msg)

st.sidebar.subheader("Animation")
designs = [
    "Green shades with orange markers",
    "Brown shades with green markers",
    "Blue shades with  orange markers",
    "Grey shades with black brown markers",
]
selected_cmap_index = st.sidebar.selectbox("Colour Scheme",
                                           range(len(designs)),
                                           format_func=lambda i: designs[i],
                                           index=0)

if dimension == 1:
    animation_type = "1d"
elif dimension == 2:
    animation_type = "2d"

if dimension is not None:
    run_disabled = False
else:
    if bounds is None:
        run_disabled = True
    elif len(bounds) == 0:
        run_disabled = True
    else:
        run_disabled = False

if dimension is None and bounds is not None:
    if len(bounds) == 1:
        animation_type = "1d"
    elif len(bounds) == 2:
        animation_type = "2d"
    else:
        available_plots = ["Contour", "PCA"]
        animation_type = st.sidebar.selectbox("Plot Type", available_plots)
        if animation_type == "Contour" and len(bounds) > 2:
            st.sidebar.subheader("Fixed Values for Other Dimensions")
            fixed_values = []
            for i in range(2, len(bounds)):
                min_val, max_val = bounds[i]
                fixed_value = st.sidebar.number_input(
                    f"Value for Dimension {i + 1}",
                    min_value=min_val,
                    max_value=max_val,
                    value=0,
                    key=f"fixed_{i}",
                )
                fixed_values.append(fixed_value)

if st.sidebar.button("RUN PSO",
                     use_container_width=True,
                     disabled=run_disabled):
    try:
        response = run_pso(
            topology=topology,
            problem_type=problem_type,
            num_particles=num_particles,
            max_iterations=max_iterations,
            bounds=bounds,
            cognitive_weight=cognitive_weight,
            social_weight=social_weight,
            inertia_weight=inertia_weight,
            objective_function=chosen_function,
        )
        run_id = response["run_id"]
    except KeyError as e:
        st.error(f"Error: {e}")

    st.subheader("Summary Metrics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Time Taken (s)",
                  round(response.get("time_taken_seconds"), decimal_places))
    with col2:
        st.metric("Converged", response.get("converged"))
    with col3:
        heading = "Iteration" if convergence_type == "Iteration" else "Rate"
        if convergence_type == "Iteration":
            conv_value = response.get("convergence_iteration")
        else:
            conv_value = str(
                round(response.get("convergence_rate"), decimal_places)) + "%"
        st.metric(f"Convergence {heading}", conv_value)

    st.write("""
        This section provides information about the convergence of the PSO algorithm.
        Convergence is determined based on the absolute difference between the global best score
        and the optimal score. If this difference is below a specific tolerance level,
        the algorithm is considered to have converged.

        ### Convergence Iteration vs. Convergence Rate/Proportion
        - **Convergence Iteration**: Represents the exact iteration at which the swarm's global best solution meets
        the convergence criteria. It provides a straightforward measure of how quickly the swarm converged.
        - **Convergence Rate/Proportion**: Represents the convergence iteration as a percentage of the total number
        of iterations. It allows for comparison between runs with different total iterations, where a lower percentage
        indicates faster convergence.

        You can choose between these two metrics in the settings to view the one that best fits your analysis needs.
    """)

    st.subheader("Global Best Solutions")
    st.write("The global best score and position achieved by the swarm.")
    global_values = {
        "Run ID": run_id,
        "Global Best Score": response.get("global_best_score"),
        "Global Best Position": response.get("global_best_position"),
    }
    st.write(global_values)

    st.subheader("Precision Analysis")
    st.write("""
        Precision is a measure of how close the solution is to the optimal value.
        It's represented both in terms of score and position. Lower values indicate
        higher precision.
    """)
    score_precision = response.get("score_precision")
    position_precision = response.get("position_precision")

    col5, col6 = st.columns(2)

    with col5:
        st.subheader("Score based Precision")
        st.line_chart(score_precision, use_container_width=True)

    with col6:
        st.subheader("Position based Precision")
        st.line_chart(position_precision, use_container_width=True)

    st.subheader("Visualization")
    st.write(
        "Watch the animation below to see how the PSO algorithm evolves over iterations.",
    )
    with st.spinner("Generating animation..."):
        animation_file_path = animate_pso(
            run_id=run_id,
            animation_type=animation_type,
            design_index=selected_cmap_index,
            fixed_values=fixed_values if animation_type == "Contour" else None,
        )
        st.video(animation_file_path)
