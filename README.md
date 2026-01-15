# Shopper-Spectrum-Customer-Segmentation-Product-Recommendation
# 🛒 Shopper Spectrum  
### Customer Segmentation & Product Recommendation in E-Commerce

Shopper Spectrum is an end-to-end **E-Commerce Analytics and Machine Learning project** that analyzes customer transaction data to **segment customers using RFM analysis** and **recommend products using item-based collaborative filtering**.  
The project includes a **fully interactive Streamlit web application** for real-time predictions and insights.

---

## 📌 Project Overview

The global e-commerce industry generates massive volumes of transactional data. Extracting meaningful insights from this data helps businesses improve **customer engagement, personalization, and revenue growth**.

This project focuses on:
- Understanding customer purchasing behavior
- Segmenting customers into meaningful groups
- Recommending relevant products using similarity-based methods
- Presenting insights through an interactive web application

---

## 🎯 Problem Statement

Analyze online retail transaction data to:
1. Segment customers based on **Recency, Frequency, and Monetary (RFM)** metrics
2. Identify **high-value, regular, occasional, and at-risk customers**
3. Build a **product recommendation system** using collaborative filtering
4. Provide **real-time predictions** via a Streamlit web application

---

## 🧠 Problem Type
- **Unsupervised Machine Learning** – Customer Clustering
- **Collaborative Filtering** – Product Recommendation System

---

## 🏢 Real-World Business Use Cases
- Targeted marketing campaigns  
- Personalized product recommendations  
- Customer retention and churn reduction  
- Inventory planning and demand forecasting  
- Revenue optimization through customer insights  

---

## 📂 Dataset Description

**Source:** Online Retail Transaction Dataset  

| Column | Description |
|------|-------------|
| InvoiceNo | Unique transaction ID |
| StockCode | Product code |
| Description | Product name |
| Quantity | Quantity purchased |
| InvoiceDate | Date & time of transaction |
| UnitPrice | Price per unit |
| CustomerID | Unique customer identifier |
| Country | Customer country |

---

## 🛠️ Data Preprocessing
- Removed missing `CustomerID` values  
- Excluded cancelled transactions (`InvoiceNo` starting with `C`)  
- Removed zero or negative quantity and price records  
- Converted `InvoiceDate` to datetime (day-first format)  
- Created `TotalAmount = Quantity × UnitPrice`  

---

## 📊 Exploratory Data Analysis (EDA)
- Country-wise revenue analysis  
- Top-selling products  
- Monthly sales trends  
- Transaction value distributions  
- Customer purchasing behavior patterns  

---

## 📐 RFM Analysis
- **Recency:** Days since last purchase  
- **Frequency:** Number of purchases  
- **Monetary:** Total spend  

RFM values were standardized and used as input features for clustering.

---

## 🔍 Customer Segmentation
- Algorithm: **KMeans Clustering**
- Cluster evaluation:
  - Elbow Method
  - Silhouette Score
- Customer segments:
  - 💎 High-Value Customers
  - 🙂 Regular Customers
  - 🛍️ Occasional Shoppers
  - ⚠️ At-Risk Customers

---

## 🔮 Product Recommendation System
- Approach: **Item-Based Collaborative Filtering**
- Technique:
  - Customer–Product pivot table
  - Cosine similarity between products
- Output:
  - Top 5 similar products for a given product name

---

## 🌐 Streamlit Web Application

### 🔹 Modules
1. **Home**
   - Project overview and key metrics
2. **Customer Segmentation**
   - Input RFM values
   - Predict customer segment in real time
3. **Product Recommendation**
   - Enter a product name
   - Get 5 similar product recommendations
4. **Analytics Dashboard**
   - Country-wise revenue chart
   - Monthly sales trend visualization

---

## 📁 Project Structure
Shopper-Spectrum/
│
├── app.py # Streamlit application
├── online_retail.csv # Dataset
├── kmeans_rfm_model.pkl # Trained clustering model
├── rfm_scaler.pkl # StandardScaler for RFM values
├── requirements.txt # Project dependencies
└── README.md # Project documentation


---
📌 Conclusion

Shopper Spectrum demonstrates how data science and machine learning can transform raw e-commerce data into actionable business insights. The project integrates analytics, modeling, and deployment into a single, scalable solution suitable for real-world applications.

👤 Author

Samagra Gupta
Data Science / AIML Project

## ⚙️ Installation & Setup

### 🔹 Install Dependencies
```bash
pip install -r requirements.txt


