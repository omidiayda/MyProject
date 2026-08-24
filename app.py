import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from io import StringIO

# تنظیمات صفحه داشبورد
st.set_page_config(
    page_title="داشبورد پیش‌بینی عملکرد تحصیلی", 
    page_icon="🎓", 
    layout="wide"
)

# بارگذاری مدل و ویژگی‌ها
@st.cache_resource
def load_model_and_features():
    model = joblib.load('student_performance_model.pkl')
    features = joblib.load('model_features.pkl')
    return model, features

try:
    model, feature_columns = load_model_and_features()
except Exception as e:
    st.error(f"❌ خطا در بارگذاری فایل‌های مدل: {e}")
    st.info("💡 لطفاً فایل‌های `student_performance_model.pkl` و `model_features.pkl` را در پوشه پروژه قرار دهید.")
    st.stop()

st.title("🎓 سامانه هوشمند پیش‌بینی عملکرد تحصیلی دانشجویان")
st.markdown("""
این اپلیکیشن بر اساس الگوریتم **Random Forest** وضعیت عملکرد تحصیلی دانشجو را پیش‌بینی می‌کند.
""")

# تب‌های مختلف
tab1, tab2, tab3 = st.tabs(["🔮 پیش‌بینی تک نفره", "📊 تحلیل گروهی", "ℹ️ راهنما"])

with tab1:
    st.sidebar.header("📝 ورود اطلاعات دانشجو")
    
    def user_input_features():
        # دریافت ویژگی‌های اصلی
        sex = st.sidebar.selectbox("جنسیت", ['زن (F)', 'مرد (M)'])
        age = st.sidebar.slider("سن", 15, 22, 17)
        address = st.sidebar.selectbox("نوع سکونت", ['شهری (U)', 'روستایی (R)'])
        
        G1 = st.sidebar.slider("نمره ترم اول (G1)", 0.0, 20.0, 12.0, step=0.5)
        G2 = st.sidebar.slider("نمره ترم دوم (G2)", 0.0, 20.0, 13.0, step=0.5)
        
        studytime = st.sidebar.selectbox(
            "زمان مطالعه هفتگی", 
            [1, 2, 3, 4], 
            format_func=lambda x: {1: '<2 ساعت', 2: '2-5 ساعت', 3: '5-10 ساعت', 4: '>10 ساعت'}[x]
        )
        failures = st.sidebar.selectbox("تعداد مردودی‌های قبلی", [0, 1, 2, 3])
        absences = st.sidebar.slider("تعداد غیبت‌ها", 0, 93, 4)
        
        higher = st.sidebar.selectbox("تمایل به تحصیلات عالی", ['بله', 'خیر'])
        internet = st.sidebar.selectbox("دسترسی به اینترنت", ['بله', 'خیر'])
        
        famsize = st.sidebar.selectbox("اندازه خانواده", ['کوچک (LE3)', 'بزرگ (GT3)'])
        Pstatus = st.sidebar.selectbox("وضعیت زندگی والدین", ['با هم', 'جدا شده'])
        
        # ساخت دیکشنری با مقادیر منطقی‌تر
        data = {}
        for col in feature_columns:
            if col in ['sex', 'address', 'higher', 'internet', 'famsize', 'Pstatus']:
                data[col] = 0  # مقدار پیش‌فرض برای متغیرهای categorical
            elif col in ['age', 'studytime', 'failures', 'absences', 'G1', 'G2']:
                data[col] = 0  # بعداً جایگزین می‌شود
            else:
                data[col] = 0
        
        # پر کردن مقادیر واقعی
        data['sex'] = 0 if sex == 'زن (F)' else 1
        data['age'] = age
        data['address'] = 1 if address == 'شهری (U)' else 0
        data['G1'] = G1
        data['G2'] = G2
        data['studytime'] = studytime
        data['failures'] = failures
        data['absences'] = absences
        data['higher'] = 1 if higher == 'بله' else 0
        data['internet'] = 1 if internet == 'بله' else 0
        data['famsize'] = 0 if famsize == 'کوچک (LE3)' else 1
        data['Pstatus'] = 0 if Pstatus == 'با هم' else 1
        
        features_df = pd.DataFrame([data], columns=feature_columns)
        return features_df
    
    input_df = user_input_features()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📋 اطلاعات وارد شده:")
        st.dataframe(input_df.T, use_container_width=True)
    
    with col2:
        st.subheader("🎯 نتیجه پیش‌بینی")
        if st.button("🚀 پیش‌بینی کن", type="primary", use_container_width=True):
            prediction = model.predict(input_df)[0]
            prediction_proba = model.predict_proba(input_df)[0]
            
            st.markdown("---")
            
            # نمایش احتمال با نمودار
            fig = go.Figure(data=[go.Pie(
                labels=['قبولی', 'مردودی'],
                values=[prediction_proba[1], prediction_proba[0]],
                hole=.4,
                marker_colors=['#2ecc71', '#e74c3c']
            )])
            fig.update_layout(showlegend=False, height=200)
            st.plotly_chart(fig, use_container_width=True)
            
            if prediction == 1:
                st.success(f"### ✅ قبولی")
                st.metric("احتمال موفقیت", f"{prediction_proba[1]*100:.1f}%")
            else:
                st.error(f"### ❌ در معرض خطر")
                st.metric("احتمال مردودی", f"{prediction_proba[0]*100:.1f}%")
                st.warning("⚠️ نیاز به مداخله آموزشی دارد")

with tab2:
    st.header("📊 تحلیل گروهی دانشجویان")
    uploaded_file = st.file_uploader("فایل CSV دانشجویان را آپلود کنید", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ {len(df)} رکورد بارگذاری شد")
            
            # پیش‌پردازش مشابه تک‌نفره
            # (باید مطابق با آموزش مدل باشد)
            
            predictions = model.predict(df[feature_columns])
            probabilities = model.predict_proba(df[feature_columns])
            
            df['پیش‌بینی'] = ['قبولی' if p == 1 else 'مردودی' for p in predictions]
            df['احتمال_قبولی'] = probabilities[:, 1] * 100
            
            st.subheader("نتایج پیش‌بینی:")
            st.dataframe(df[['پیش‌بینی', 'احتمال_قبولی']], use_container_width=True)
            
            # آمار کلی
            col1, col2, col3 = st.columns(3)
            col1.metric("تعداد کل", len(df))
            col2.metric("تعداد قبولی", (predictions == 1).sum())
            col3.metric("تعداد مردودی", (predictions == 0).sum())
            
        except Exception as e:
            st.error(f"خطا در پردازش فایل: {e}")

with tab3:
    st.header("ℹ️ راهنمای استفاده")
    st.markdown("""
    ### ویژگی‌های مهم مدل:
    - **نمرات ترم‌های قبل (G1, G2)**: مهم‌ترین فاکتور پیش‌بینی
    - **زمان مطالعه**: تأثیر مستقیم بر موفقیت
    - **تعداد غیبت‌ها**: رابطه معکوس با عملکرد
    - **سابقه مردودی**: ریسک فاکتور قوی
    
    """)

# فوتر
st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>طراحی شده با Streamlit و Random Forest</p>", unsafe_allow_html=True)