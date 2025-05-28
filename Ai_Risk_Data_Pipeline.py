# (여기에 최신 코드가 삽입됩니다)
# AI 버블 리스크 예측을 위한 데이터 수집 파이프라인 및 예측 모델 프로토타입
# 필요한 라이브러리 불러오기
from statsmodels.tsa.stattools import grangercausalitytests
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from fpdf import FPDF
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from email.mime.text import MIMEText
import matplotlib.pyplot as plt
import schedule
import time
import os
import streamlit as st
import openai
import yfinance as yf
import requests
import telegram
import datetime
import pandas as pd
import seaborn as sns
from scipy.stats import sem, t
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from bs4 import BeautifulSoup
import json
from streamlit_autorefresh import st_autorefresh
import pydeck as pdk
import sqlite3

# 사용자 이메일 등록 기능
USER_EMAILS_DB = "user_emails.db"
def init_email_db():
    conn = sqlite3.connect(USER_EMAILS_DB)
    c = conn.cursor()
    c.execute("CREATE TABLE IF NOT EXISTS emails (email TEXT PRIMARY KEY)")
    conn.commit()
    conn.close()

def save_email(email):
    conn = sqlite3.connect(USER_EMAILS_DB)
    c = conn.cursor()
    c.execute("INSERT OR IGNORE INTO emails (email) VALUES (?)", (email,))
    conn.commit()
    conn.close()

def get_all_emails():
    conn = sqlite3.connect(USER_EMAILS_DB)
    c = conn.cursor()
    c.execute("SELECT email FROM emails")
    emails = [row[0] for row in c.fetchall()]
    conn.close()
    return emails

# 대시보드에서 사용자 이메일 입력
with st.sidebar:
    st.markdown("## 📧 이메일 등록")
    user_email_input = st.text_input("보고서를 받을 이메일 주소 입력")
    if st.button("이메일 저장"):
        if user_email_input:
            save_email(user_email_input)
            st.success(f"{user_email_input} 이메일이 등록되었습니다.")

# 사용자 전체에게 이메일 전송 기능
def send_email_report_to_all(filename):
    recipients = get_all_emails()
    for email in recipients:
        send_email_report(email, filename)

# 위험도 기반 자동 리밸런싱 로직 (시뮬레이션용)
def simulate_portfolio_adjustment(cri):
    weights = {
        "현금": 0.6 if cri > 0.7 else 0.4 if cri > 0.5 else 0.2,
        "채권": 0.3 if cri > 0.7 else 0.4 if cri > 0.5 else 0.5,
        "주식": 0.1 if cri > 0.7 else 0.2 if cri > 0.5 else 0.3,
    }
    return weights

# 리밸런싱 결과 시각화
def show_portfolio_adjustment(weights):
    st.subheader("💼 리스크 기반 포트폴리오 비중 조정")
    fig, ax = plt.subplots()
    ax.pie(weights.values(), labels=weights.keys(), autopct='%1.1f%%')
    st.pyplot(fig)

# 포트폴리오 시뮬레이션 실행
adjusted_weights = simulate_portfolio_adjustment(latest_cri)
show_portfolio_adjustment(adjusted_weights)

# 전체 이메일 수신자에게 보고서 발송
send_email_report_to_all("cri_report.pdf")
