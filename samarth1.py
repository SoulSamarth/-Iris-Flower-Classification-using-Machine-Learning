


import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

st.set_page_config(page_title="Iris Dataset Analysis", layout="wide")
st.title("🌸 Iris Dataset Analysis and Classification")

# Load dataset
df = pd.read_csv("Iris.csv")
st.write("## Dataset Preview")
st.dataframe(df)

# Data Preparation
st.write("## Data Description")
st.write(df.describe())

st.write("## Class Distribution")
st.bar_chart(df['Species'].value_counts())

st.write("## Data Visualization")
col1, col2 = st.columns(2)

with col1:
    st.write("### Sepal Length vs Sepal Width")
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    sns.scatterplot(data=df, x="SepalLengthCm", y="SepalWidthCm", hue="Species", ax=ax1)
    st.pyplot(fig1)
    st.markdown("""
    **Insight:** The scatter plot shows the relationship between Sepal Length and Sepal Width across different species. Iris-setosa generally has smaller sepal length but larger sepal width. Iris-versicolor and Iris-virginica overlap more, indicating these features alone may not perfectly separate them.

    **Interpretation:** Iris-setosa is linearly separable using these features, but others require additional dimensions to distinguish.
    """)

with col2:
    st.write("### Petal Length vs Petal Width")
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    sns.scatterplot(data=df, x="PetalLengthCm", y="PetalWidthCm", hue="Species", ax=ax2)
    st.pyplot(fig2)
    st.markdown("""
    **Insight:** This graph provides clear separability between all three Iris species. Petal dimensions are highly predictive. Iris-setosa clusters distinctly, while versicolor and virginica form separate but slightly overlapping clusters.

    **Interpretation:** Petal dimensions are more informative for classification than sepal dimensions.
    """)

st.write("### Pairplot")
fig3 = sns.pairplot(df.drop("Id", axis=1), hue="Species", height=2.0)
st.pyplot(fig3)
st.markdown("""
**Insight:** Pairplots show pairwise feature distributions. It is evident that petal measurements give better class separability than sepal measurements.

**Conclusion from pairplot:** Strong correlation exists between petal length and width. This reinforces their utility in predicting species.
""")

st.write("### Correlation Heatmap")
fig4, ax4 = plt.subplots(figsize=(6, 4))
sns.heatmap(df.drop(["Id", "Species"], axis=1).corr(), annot=True, cmap="coolwarm", ax=ax4)
st.pyplot(fig4)
st.markdown("""
**Insight:** Petal Length and Petal Width have a strong positive correlation (~0.96), which is a good indicator for classification. Sepal features are moderately correlated.

**Conclusion from heatmap:** Feature engineering should focus on petal-based attributes for better model accuracy.
""")

# Additional Visualizations
col3, col4 = st.columns(2)

with col3:
    st.write("### Boxplot by Species")
    fig6, ax6 = plt.subplots(figsize=(6, 4))
    sns.boxplot(x="Species", y="PetalLengthCm", data=df, ax=ax6)
    st.pyplot(fig6)
    st.markdown("""
    **Insight:** The boxplot shows Petal Length distribution for each species. Iris-setosa has a clearly distinct range of petal lengths compared to the other two species. Virginica shows the highest spread.

    **Interpretation:** Petal length alone provides strong evidence for classifying Iris-setosa.
    """)

with col4:
    st.write("### Violin Plot of Sepal Width by Species")
    fig7, ax7 = plt.subplots(figsize=(6, 4))
    sns.violinplot(x="Species", y="SepalWidthCm", data=df, ax=ax7)
    st.pyplot(fig7)
    st.markdown("""
    **Insight:** Violin plots combine boxplots and KDEs, showing both distribution and range. Sepal Width for setosa tends to be wider and less varied compared to other species.

    **Conclusion:** Although not as distinct as petal features, Sepal Width helps in identifying setosa.
    """)

# Model Training
st.write("## Model Training and Evaluation")
X = df.drop(["Id", "Species"], axis=1)
y = df["Species"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
st.write(f"**Model Accuracy:** {acc*100:.2f}%")
st.text("Classification Report:")
st.text(classification_report(y_test, y_pred))
st.write("### Confusion Matrix")
fig5, ax5 = plt.subplots(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y), yticklabels=np.unique(y), ax=ax5)
plt.xlabel("Predicted")
plt.ylabel("Actual")
st.pyplot(fig5)

# Prediction Interface
st.write("## Predict Iris Species")
sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.8)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.0)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 4.3)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 1.3)

input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
prediction = model.predict(input_data)[0]
st.success(f"Predicted Iris Species: **{prediction}**")

# Final Conclusion
st.write("## Final Conclusion")
st.markdown("""
This analysis revealed that petal-based features (PetalLengthCm and PetalWidthCm) are the most important in classifying Iris species, as evidenced by the clear clustering and high correlation in visualizations. Additional plots such as boxplots and violin plots reinforce these findings, demonstrating the utility of Petal Length in separating species.

Decision Tree Classifier achieved high accuracy (~97%) with just a max depth of 3, underscoring how well-defined the classes are within this dataset. Visualizations like pairplots and correlation heatmaps helped identify the best features for classification. This project bridges theoretical knowledge with hands-on practical skills including data visualization, preprocessing, model building, and evaluation, while showing that simple interpretable models can perform well with good data.
""")