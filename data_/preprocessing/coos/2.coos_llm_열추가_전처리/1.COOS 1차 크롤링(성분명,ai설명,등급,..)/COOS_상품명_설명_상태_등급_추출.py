import numpy as np
import pandas as pd
import re
import requests
from bs4 import BeautifulSoup
import time
import urllib.parse
import os
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
#################################성분 추출 & 전처리 
## OLIVEYOUNG에서 성분추출
def ingredient_from_oy(path):
    df = pd.read_csv(path)
    ing = df['성분']
    return ing 

## 성분 개별행 만들기 
def unique_ingredients(ing):
    unique_ing = (ing
                  .str.split(', ') #콤마,공백으로 분리
                  .explode() # 리스트 -> 개별행
                  .str.strip() # 앞뒤 공백 제거
                  .unique())
    return unique_ing

## 전처리.v1
def preprocess_ing(unique_ing):
    #문자열로 만들기
    if isinstance(unique_ing, (list,tuple)) or hasattr(unique_ing, 'dtype'): #리스트,튜플,numpy배열, pandas Series인지 확인
        unique_ing = ', '.join(str(item) for item in unique_ing)
    else:
        unique_ing = str(unique_ing) # 1아니고, 문자열,숫자등 인경우 
        
    #1단계 : 화학명 쉼표 보호 ex : (1,2-헥산다이올)
    processed = re.sub(r'(\d+),(\d+)-',r'\1§\2-',unique_ing)
    
    #2단계 : 구분자 기준 분리
    separators = r'\[\s*\]|,| {2,}'
    ingredients = re.split(separators, processed)

    # 3단계: 쉼표 복원
    ingredients = [re.sub(r'§', ',', ing) for ing in ingredients]

    # 4단계: 부가정보 및 "제공된 이후" 제거
    cleaned = []
    for ing in ingredients:
        ing = re.sub(r'제공된.*', '', ing)  # '제공된'부터 끝까지 제거
        ing = re.sub(r'\s*\([^()]*?(ppm|mg|%|ppb)\)', '', ing)  # 괄호 단위 제거
        ing = re.sub(r'[+]+', '', ing)  # +, ++ 제거
        ing = re.sub(r'\s*\*+', '', ing)   # * 제거 5 

        ing = re.sub(r'[()]', '', ing)  # 남은 괄호 제거
        ing = re.sub(r'\([^()\uAC00-\uD7A3]*\)', '', ing)

        cleaned.append(ing.strip())

    # 5단계: 빈 문자열 제거 및 중복 제거
    final = list(dict.fromkeys([item for item in cleaned if item]))

    return final


################################# URL 크롤링
# 성분 정보 수집 함수 (Selenium 기반)
def get_ingredient_description_selenium(ingredient_name):
    options = Options()
    options.add_argument('--headless')
    driver = webdriver.Chrome(options=options)
    
    try:
        encoded_ingredient = urllib.parse.quote(ingredient_name, safe='')
        url = f'https://coos.kr/ingredients/{encoded_ingredient}'
        driver.get(url)
        time.sleep(2)
        
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        
        # AI 설명
        description = "설명 없음"
        ai_desc_div = soup.find('div', string='AI Description')
        if ai_desc_div:
            description_p = ai_desc_div.find_next('p')
            if description_p:
                description = description_p.get_text(strip=True)

        # 등급 정보
        label = None
        score = None
        all_tds = soup.find_all('td', class_='MuiTableCell-root')
        for td in all_tds:
            span = td.find('span')
            number_div = td.find('div', style=lambda x: x and 'background' in x)
            if span and number_div:
                label = span.get_text(strip=True)
                score = number_div.get_text(strip=True)

        return description, label, score
    
    except Exception as e:
        print(f"오류 ({ingredient_name}): {e}")
        return "오류", "오류", "오류"
    
    finally:
        driver.quit()

########################################
# 성분 리스트에서 정보 수집 및 DataFrame 저장
def create_df_from_korean_name(csv_path):
    df = pd.read_csv(csv_path)
    ingredients = df['korean_name'].dropna().unique().tolist()
    
    print(f"총 {len(ingredients)}개 성분 정보 수집 시작...\n")

    data = []
    for i, name in enumerate(ingredients, 1):
        print(f"[{i}/{len(ingredients)}] {name} 처리 중...")
        desc, label, score = get_ingredient_description_selenium(name)
        data.append({
            '성분명': name,
            'AI_설명': desc,
            '상태': label,
            '등급': score
        })
        print(f"설명: {desc} / 상태: {label} / 등급: {score}")
        print("-" * 40)
        time.sleep(0.5)

    result_df = pd.DataFrame(data)
    return result_df

########################################
# 실행 부분
if __name__ == "__main__":
    input_csv = "/Users/sein/Desktop/programming/portfolio/projects/화장품/cooscrawling/COOS_성분_통합.csv"
    output_csv = "/Users/sein/Desktop/programming/portfolio/projects/화장품/cooscrawling/COOS_성분_상세정보.csv"

    result_df = create_df_from_korean_name(input_csv)
    result_df.to_csv(output_csv, index=False, encoding='utf-8-sig')

    print(f"\n📁 수집된 정보가 다음 위치에 저장되었습니다: {output_csv}")