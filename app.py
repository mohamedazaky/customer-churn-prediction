"""Customer churn dashboard and predictor (Streamlit)."""
import json

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from churn_utils import FEATURES, METRICS_PATH, MODEL_PATH, load_data

st.set_page_config(page_title="Customer Churn Analytics", page_icon="📉", layout="wide")

STAY, LEAVE = "#1F6F8B", "#D1495B"
PLOT_LAYOUT = dict(margin=dict(l=10, r=10, t=10, b=10), height=320,
                   plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")


@st.cache_data
def get_data() -> pd.DataFrame:
    return load_data()


def ensure_model() -> None:
    """Train the model on first run so the app works straight after cloning."""
    if not MODEL_PATH.exists() or not METRICS_PATH.exists():
        import train
        with st.spinner("Training the model for the first time (about 20 seconds)..."):
            train.main()


@st.cache_resource
def get_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def get_metrics() -> dict:
    return json.loads(METRICS_PATH.read_text())


def churn_rate_by(df: pd.DataFrame, col: str) -> pd.DataFrame:
    out = df.groupby(col, observed=True)["Churn"].agg(["mean", "size"]).reset_index()
    out.columns = [col, "Churn rate", "Customers"]
    return out


def rate_bar(df: pd.DataFrame, col: str, horizontal: bool = False, keep_order: bool = False):
    data = churn_rate_by(df, col)
    if not keep_order:
        data = data.sort_values("Churn rate")
    kwargs = dict(x="Churn rate", y=col, orientation="h") if horizontal else dict(x=col, y="Churn rate")
    fig = px.bar(data, text_auto=".0%", hover_data=["Customers"], **kwargs)
    fig.update_traces(marker_color=LEAVE, textposition="outside", cliponaxis=False)
    axis = "xaxis" if horizontal else "yaxis"
    fig.update_layout(**PLOT_LAYOUT, **{axis: dict(tickformat=".0%", title=None)})
    fig.update_layout(**{("yaxis" if horizontal else "xaxis"): dict(title=None)})
    return fig


ensure_model()
df = get_data()
metrics = get_metrics()

with st.sidebar:
    st.markdown("### Customer Churn Analytics")
    st.caption("Find which customers are likely to cancel, and why.")
    page = st.radio("Go to", ["Overview", "Churn drivers", "Model performance", "Predict a customer"],
                    label_visibility="collapsed")
    st.divider()
    st.caption(f"Dataset: {len(df):,} telecom customers, {len(FEATURES)} attributes each.")


# ---------------------------------------------------------------- Overview
if page == "Overview":
    st.title("Overview")
    churned = df[df["Churn"] == 1]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers", f"{len(df):,}")
    c2.metric("Churn rate", f"{df['Churn'].mean():.1%}")
    c3.metric("Monthly revenue lost", f"${churned['MonthlyCharges'].sum():,.0f}")
    c4.metric("Avg. tenure of churners", f"{churned['tenure'].mean():.0f} months")

    left, right = st.columns(2)
    with left:
        st.subheader("Churn rate by contract")
        st.plotly_chart(rate_bar(df, "Contract"), width="stretch")
    with right:
        st.subheader("Churn rate by tenure")
        st.plotly_chart(rate_bar(df, "TenureGroup", keep_order=True), width="stretch")

    st.subheader("Monthly charges: customers who stayed vs. left")
    hist = df.assign(Status=df["Churn"].map({0: "Stayed", 1: "Left"}))
    fig = px.histogram(hist, x="MonthlyCharges", color="Status", nbins=40, barmode="overlay",
                       opacity=0.75, color_discrete_map={"Stayed": STAY, "Left": LEAVE})
    fig.update_layout(**PLOT_LAYOUT, xaxis_title="Monthly charges ($)", yaxis_title="Customers",
                      legend_title=None)
    st.plotly_chart(fig, width="stretch")


# ---------------------------------------------------------- Churn drivers
elif page == "Churn drivers":
    st.title("Churn drivers")
    st.caption("Filter the customer base, then compare churn rates across any attribute.")

    f1, f2, f3 = st.columns(3)
    contract = f1.multiselect("Contract", sorted(df["Contract"].unique()))
    internet = f2.multiselect("Internet service", sorted(df["InternetService"].unique()))
    senior = f3.selectbox("Senior citizen", ["All", "Yes", "No"])

    view = df.copy()
    if contract:
        view = view[view["Contract"].isin(contract)]
    if internet:
        view = view[view["InternetService"].isin(internet)]
    if senior != "All":
        view = view[view["SeniorCitizen"] == senior]

    if view.empty:
        st.warning("No customers match these filters. Remove a filter to see results.")
        st.stop()

    m1, m2 = st.columns(2)
    m1.metric("Customers in selection", f"{len(view):,}")
    m2.metric("Churn rate in selection", f"{view['Churn'].mean():.1%}",
              delta=f"{(view['Churn'].mean() - df['Churn'].mean()) * 100:+.1f} pts vs. all",
              delta_color="inverse")

    options = ["PaymentMethod", "InternetService", "TechSupport", "OnlineSecurity",
               "PaperlessBilling", "Dependents", "Partner", "StreamingTV"]
    attr = st.selectbox("Compare churn rate by", options)
    st.plotly_chart(rate_bar(view, attr, horizontal=True), width="stretch")

    with st.expander("See customers in this selection"):
        st.dataframe(view[["customerID", "tenure", "Contract", "InternetService",
                           "MonthlyCharges", "Churn"]], width="stretch", hide_index=True)


# ------------------------------------------------------ Model performance
elif page == "Model performance":
    st.title("Model performance")
    best = metrics["best_model"]
    st.caption(f"Three models were trained on {metrics['n_train']:,} customers and tested on "
               f"{metrics['n_test']:,} unseen customers. {best} was selected for the best F1 score.")

    res = metrics["comparison"][best]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ROC-AUC", f"{res['roc_auc']:.3f}")
    c2.metric("Recall", f"{res['recall']:.0%}", help="Share of real churners the model catches.")
    c3.metric("Precision", f"{res['precision']:.0%}", help="Share of flagged customers who really churn.")
    c4.metric("F1 score", f"{res['f1']:.3f}")

    left, right = st.columns(2)
    with left:
        st.subheader("What drives the prediction")
        imp = pd.Series(metrics["feature_importance"]).head(10).sort_values()
        fig = px.bar(x=imp.values, y=imp.index, orientation="h", text_auto=".0%")
        fig.update_traces(marker_color=STAY)
        fig.update_layout(**PLOT_LAYOUT, xaxis=dict(tickformat=".0%", title=None), yaxis_title=None)
        st.plotly_chart(fig, width="stretch")
    with right:
        st.subheader("Confusion matrix")
        cm = res["confusion_matrix"]
        fig = go.Figure(go.Heatmap(z=cm, x=["Predicted stay", "Predicted leave"],
                                   y=["Actually stayed", "Actually left"], text=cm,
                                   texttemplate="%{text}", colorscale=[[0, "#E6ECF0"], [1, STAY]],
                                   showscale=False))
        fig.update_layout(**PLOT_LAYOUT)
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig, width="stretch")

    st.subheader("Model comparison")
    table = pd.DataFrame(metrics["comparison"]).T.drop(columns="confusion_matrix").astype(float)
    st.dataframe(table.style.format("{:.3f}"), width="stretch")


# ----------------------------------------------------- Predict a customer
else:
    st.title("Predict a customer")
    st.caption("Enter a customer's details to estimate how likely they are to cancel.")
    result_area = st.container(border=True)
    st.markdown("#### Customer details")

    col1, col2, col3 = st.columns(3)
    with col1:
        tenure = st.slider("Tenure (months)", 0, 72, 6)
        monthly = st.number_input("Monthly charges ($)", 15.0, 120.0, 85.0, step=5.0)
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        payment = st.selectbox("Payment method", sorted(df["PaymentMethod"].unique()))
        paperless = st.radio("Paperless billing", ["Yes", "No"], horizontal=True)
    with col2:
        internet = st.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
        no_net = "No internet service"
        addon = (lambda label: no_net if internet == "No"
                 else st.radio(label, ["No", "Yes"], horizontal=True))
        security = addon("Online security")
        backup = addon("Online backup")
        protection = addon("Device protection")
        support = addon("Tech support")
    with col3:
        tv = addon("Streaming TV")
        movies = addon("Streaming movies")
        phone = st.radio("Phone service", ["Yes", "No"], horizontal=True)
        lines = "No phone service" if phone == "No" else st.radio("Multiple lines", ["No", "Yes"], horizontal=True)
        senior = st.radio("Senior citizen", ["No", "Yes"], horizontal=True)
        partner = st.radio("Has partner", ["No", "Yes"], horizontal=True)
        dependents = st.radio("Has dependents", ["No", "Yes"], horizontal=True)

    customer = pd.DataFrame([{
        "tenure": tenure, "MonthlyCharges": monthly, "TotalCharges": tenure * monthly,
        "gender": "Male", "SeniorCitizen": senior, "Partner": partner, "Dependents": dependents,
        "PhoneService": phone, "MultipleLines": lines, "InternetService": internet,
        "OnlineSecurity": security, "OnlineBackup": backup, "DeviceProtection": protection,
        "TechSupport": support, "StreamingTV": tv, "StreamingMovies": movies,
        "Contract": contract, "PaperlessBilling": paperless, "PaymentMethod": payment,
    }])[FEATURES]

    prob = float(get_model().predict_proba(customer)[0, 1])
    if prob >= 0.6:
        level, color, advice = "High risk", LEAVE, "Contact this customer now with a retention offer, such as a discounted annual contract."
    elif prob >= 0.35:
        level, color, advice = "Medium risk", "#E09F3E", "Keep an eye on this customer and consider adding tech support or security at a discount."
    else:
        level, color, advice = "Low risk", STAY, "No action needed. This customer is likely to stay."

    g, txt = result_area.columns([1, 1.4])
    with g:
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=prob * 100,
            number={"suffix": "%", "valueformat": ".0f", "font": {"color": color}},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": color},
                   "steps": [{"range": [0, 35], "color": "#E6ECF0"},
                             {"range": [35, 60], "color": "#F3E3C7"},
                             {"range": [60, 100], "color": "#F2D0D5"}]}))
        fig.update_layout(height=220, margin=dict(l=40, r=40, t=30, b=0),
                          paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, width="stretch")
    with txt:
        st.markdown(f"<h3 style='color:{color};margin-bottom:0'>{level}</h3>", unsafe_allow_html=True)
        st.markdown(f"Estimated churn probability: **{prob:.0%}**")
        st.write(advice)
