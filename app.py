import streamlit as st
import plotly.express as px
from predict import predict_image

st.set_page_config(
    page_title="NeuralLens",
    page_icon="🔍",
    layout="centered"
)

st.title("🔍 Neural Lens")
st.markdown("#### AI-Powered Image Recognition Engine")
st.markdown("Upload any image and our **ResNet50** deep learning model will identify what's in it!")
st.divider()

with st.sidebar:
    st.markdown("### ℹ️ About")
    st.info("""
    **NeuralLens** uses ResNet50
    trained on ImageNet dataset.
    
    It can recognize **1000+** 
    different object categories!
    """)
    st.divider()
    st.markdown("### 📊 Model Info")
    info = {
        "🏗️ Architecture": "ResNet50",
        "🎯 Top-5 Accuracy": "92%+",
        "📦 Classes": "1000",
        "🗂️ Training Data": "ImageNet"
    }
    for key, value in info.items():
        col1, col2 = st.columns([1.5, 1])
        col1.markdown(f"**{key}**")
        col2.markdown(value)
    
    st.divider()
    st.markdown("### 🖼️ Supported Formats")
    st.markdown("JPG, JPEG, PNG, WEBP")

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    uploaded_file = st.file_uploader(
        "Drop your image here",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed"
    )

if uploaded_file is not None:
    st.image(
        uploaded_file,
        caption="📸 Uploaded Image",
        use_container_width=True
    )
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        classify_btn = st.button(
            "🔍 Classify Image",
            use_container_width=True,
            type="primary"
        )

    if classify_btn:
        with st.spinner("🧠 Analyzing image through 50 layers..."):
            image_bytes = uploaded_file.read()
            results = predict_image(image_bytes)

        st.divider()

        top_label = results[0][1].replace('_', ' ').title()
        top_conf = results[0][2]

        st.markdown("### 🎯 Top Prediction")

        col1, col2 = st.columns(2)
        with col1:
            if top_conf > 0.7:
                st.success(f"## {top_label}")
                st.markdown("🟢 **High Confidence**")
            elif top_conf > 0.4:
                st.info(f"## {top_label}")
                st.markdown("🔵 **Medium Confidence**")
            else:
                st.warning(f"## {top_label}")
                st.markdown("🟡 **Low Confidence**")

        with col2:
            st.metric(
                label="Confidence Score",
                value=f"{top_conf:.2%}"
            )
            st.metric(
                label="Rank",
                value=f"#1 of 1000 classes"
            )

        st.divider()

        st.markdown("### 📊 Top 5 Predictions")

        labels = [r[1].replace('_', ' ').title() for r in results]
        confidences = [round(r[2] * 100, 2) for r in results]

        fig = px.bar(
            x=confidences,
            y=labels,
            orientation='h',
            labels={'x': 'Confidence (%)', 'y': ''},
            color=confidences,
            color_continuous_scale='Blues',
            text=[f"{c:.1f}%" for c in confidences]
        )
        fig.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            showlegend=False,
            coloraxis_showscale=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=300,
            margin=dict(l=0, r=0, t=0, b=0)
        )
        fig.update_traces(textposition='outside')
        st.plotly_chart(fig, use_container_width=True)

        st.divider()

        st.markdown("### 📋 Detailed Results")
        for rank, (id, label, conf) in enumerate(results):
            clean_label = label.replace('_', ' ').title()
            col1, col2, col3 = st.columns([1, 3, 2])
            col1.markdown(f"**#{rank+1}**")
            col2.markdown(f"{clean_label}")
            col3.progress(float(conf), text=f"{conf:.2%}")

        st.divider()
        st.success("✅ Analysis Complete!")

else:
    st.markdown("""
    <div style='text-align: center; padding: 50px; 
    border: 2px dashed #ccc; border-radius: 10px;'>
        <h3>📸 Upload an image to get started!</h3>
        <p>Supports: JPG, PNG, WEBP</p>
    </div>
    """, unsafe_allow_html=True)

st.divider()
st.caption("⚠️ For educational purposes | Powered by ResNet50 + ImageNet")