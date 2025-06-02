from googleapiclient.discovery import build
import pandas as pd
import time

API_KEY = "AIzaSyAamoKXJcnXnaBXgxjp_ji9YjOzOU35dVE"
youtube = build("youtube", "v3", developerKey=API_KEY)

# 카테고리 및 키워드 다양화
category_keywords = {
    "영화/애니메이션": ["영화 리뷰", "넷플릭스 추천", "영화 해석", "애니 유튜버", "시네마", "드라마 분석", "명작 추천"],
    "자동차/운송": ["자동차 리뷰", "전기차", "시승기", "중고차", "모터쇼", "운전 꿀팁", "자동차 정비"],
    "음악/퍼포먼스": ["KPOP", "인디뮤지션", "커버곡", "피아노 연주", "작곡법", "보컬 레슨", "음악 추천"],
    "반려동물/동물": ["강아지 브이로그", "고양이 채널", "동물병원", "펫케어", "애견 훈련", "유기동물", "파충류 유튜버"],
    "스포츠/운동": ["축구 하이라이트", "농구 유튜버", "운동 루틴", "헬스 유튜버", "야구 분석", "홈트레이닝", "스포츠 리뷰"],
    "여행/이벤트": ["국내여행", "해외여행", "여행 브이로그", "혼자 여행", "맛집 여행", "캠핑 브이로그", "여행 정보"],
    "게임/취미": ["게임 리뷰", "롤 유튜버", "마인크래프트", "모바일 게임", "공포 게임", "콘솔 게임", "게임 방송"],
    "일상/브이로그": ["일상 브이로그", "자취 브이로그", "직장인 브이로그", "감성 브이로그", "학생 브이로그", "루틴 영상", "라이프로그"],
    "코미디/유머": ["개그 유튜버", "몰카 채널", "웃긴 영상", "꿀잼 유튜버", "코믹 브이로그", "유머 콘텐츠", "짤방"],
    "연예/엔터테인먼트": ["아이돌 리액션", "연예인 클립", "예능 하이라이트", "팬 유튜버", "방송 리뷰", "연예인 인터뷰", "리액션 영상"],
    "뉴스/시사": ["정치 유튜버", "뉴스 요약", "시사 해설", "사회 이슈", "논평 채널", "시사 브리핑", "핫이슈 정리"],
    "뷰티/스타일": ["뷰티 유튜버", "메이크업 튜토리얼", "패션 스타일링", "헤어스타일", "셀프 네일", "옷 추천", "화장법"],
    "생활 노하우": ["자취 노하우", "생활 꿀팁", "셀프 인테리어", "정리 정돈", "청소법", "요리 초보", "살림 유튜버"],
    "교육/공부": ["수학 강의", "영어 공부법", "토익 공부", "공부법 공유", "입시 정보", "자격증 준비", "교육 콘텐츠"],
    "IT/기술": ["AI 유튜버", "코딩 강의", "IT 리뷰", "디지털 기기", "프로그래밍", "기술 해설", "스마트폰 꿀팁"],
    "과학/탐구": ["과학 실험", "우주 해설", "물리 유튜버", "화학 채널", "과학자 이야기", "로봇 기술", "생물 탐구"],
    "재테크/경제": ["주식 투자", "부동산 분석", "ETF 소개", "월급관리", "파이어족", "경제 뉴스", "가계부 쓰기"],
    "심리/자기계발": ["심리학 채널", "자기계발 유튜버", "멘탈 관리", "명상", "독서법", "습관 만들기", "시간관리"],
    "건강/의학": ["건강관리", "영양제 정보", "의사 유튜버", "운동과 건강", "다이어트", "병원 리뷰", "한방 치료"],
    "요리/음식": ["자취 요리", "간편식 레시피", "자주 해먹는 요리", "한식 요리법", "베이킹", "집밥 유튜버", "요리 꿀팁"],
    "비영리/사회": ["환경 보호", "사회 공헌", "기부 유튜버", "봉사활동", "청년 운동", "비영리 단체 소개", "공익 캠페인"]
}


# 채널 수집 함수
def search_channels(keyword, max_results=25):
    search_response = youtube.search().list(
        part="snippet",
        type="channel",
        q=keyword,
        maxResults=max_results,
        regionCode="KR"
    ).execute()

    channel_ids = [item["snippet"]["channelId"] for item in search_response["items"]]

    detail_response = youtube.channels().list(
        part="snippet,statistics",
        id=",".join(channel_ids)
    ).execute()

    channels = []
    for item in detail_response["items"]:
        channels.append({
            "카테고리": keyword,
            "채널명": item["snippet"]["title"],
            "채널ID": item["id"],
            "설명": item["snippet"].get("description", ""),
            "구독자 수": item["statistics"].get("subscriberCount", "비공개"),
            "영상 수": item["statistics"].get("videoCount", "비공개"),
            "총 조회수": item["statistics"].get("viewCount", "비공개")
        })
    return channels

# 전체 수집
all_channels = []
for category, keywords in category_keywords.items():
    for keyword in keywords:
        try:
            print(f"[{category}] '{keyword}' 검색 중...")
            channels = search_channels(keyword, max_results=25)
            all_channels.extend(channels)
            time.sleep(1)
        except Exception as e:
            print(f"{keyword} 오류: {e}")

# 중복 제거 및 저장
df = pd.DataFrame(all_channels).drop_duplicates(subset=["채널ID"])
df.to_csv("ML_TermProject/youtube_channels_500+.csv", index=False, encoding="utf-8-sig")
print(f"✅ 총 {len(df)}개의 유튜브 채널 저장 완료 -> 'youtube_channels_500+.csv'")
