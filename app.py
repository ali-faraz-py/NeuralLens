import streamlit as st
import plotly.express as px
from predict import predict_image



st.set_page_config(
    page_title="NeuralLens",
    page_icon="🔍",
    layout="centered"
)

st.title("🔍 NeuralLens")
st.write("Upload any image and AI will identify what's in it!")
st.divider()

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is not None:
    st.image(uploaded_file, 
             caption="Uploaded Image", 
             use_container_width=True)
    
    if st.button("🔍 Classify Image", use_container_width=True):
        with st.spinner("Analyzing image..."):
            image_bytes = uploaded_file.read()
            results = predict_image(image_bytes)

        st.divider()
        st.subheader("🎯 Results:")

        top_label = results[0][1].replace('_', ' ').title()
        top_conf = results[0][2]

        if top_conf > 0.7:
            st.success(f"### {top_label}")
        elif top_conf > 0.4:
            st.info(f"### {top_label}")
        else:
            st.warning(f"### {top_label} (low confidence)")

        st.metric("Confidence", f"{top_conf:.2%}")
        st.divider()

        st.subheader("📊 Top 5 Predictions:")

        labels = [r[1].replace('_', ' ').title() for r in results]
        confidences = [r[2] * 100 for r in results]

        fig = px.bar(
            x=confidences,
            y=labels,
            orientation='h',
            labels={'x': 'Confidence %', 'y': 'Class'},
            color=confidences,
            color_continuous_scale='Blues',
            title="Prediction Confidence"
        )
        fig.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            showlegend=False,
            coloraxis_showscale=False
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("📋 Detailed Results:")
        for rank, (id, label, conf) in enumerate(results):
            clean_label = label.replace('_', ' ').title()
            st.write(f"**{rank+1}. {clean_label}** — {conf:.2%}")

st.divider()
st.caption("⚠️ Powered by ResNet50 trained on ImageNet (1000 classes)")