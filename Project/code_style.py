st.markdown("""
    <style>
    /* Cài đặt màu nền cho phần chính */
    .main {
        background-color: #f0f2f6;
        padding: 2rem;
    }

    /* Tùy chỉnh nút bấm */
    .stButton>button {
        color: white;
        background-color: #4CAF50;
        border: none;
        padding: 12px 28px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 4px;
    }

    /* Tùy chỉnh ô nhập liệu */
    .stTextInput>div>div>input {
        padding: 10px;
        font-size: 16px;
        border-radius: 4px;
        border: 2px solid #ccc;
    }

    /* Tùy chỉnh selectbox */
    .stSelectbox>div>div>div {
        padding: 10px;
        font-size: 16px;
        border-radius: 4px;
        border: 1px solid #ccc;
    }

    /* Tùy chỉnh ô text_area */
    .stTextArea textarea {
        border: 2px solid #ccc; /* Thêm viền cho text_area */
        border-radius: 5px;
        padding: 10px;
        font-size: 16px;
    }

    /* Tùy chỉnh tiêu đề */
    h1 {
        text-align: center;
        color: #FF8C00;
        font-family: 'Georgia', serif;
    }
    </style>

    <h1> 🐝 W1 Honey Hunters 🍯 </h1>
""", unsafe_allow_html=True)

# Page title and description
st.title("📸 Image Retrieval Interface")
st.markdown("""
    Welcome to the Image Retrieval Interface. Use the form below to enter your prompt and sentence, select the retrieval method, and visualize the results.
    """)