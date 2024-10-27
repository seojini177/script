import requests
import streamlit as st
import pandas as pd
from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

# 기존 질문-답변 데이터
data = [
    {"label": 1, "question": "시적 화자와 서술자는 문학 공부에서 왜 중요한 개념인가요?", "answer": "시적 화자와 서술자는 문학에서 중요한 개념입니다. 서술자는 소설에서 이야기를 전달하는 목소리로, 작품의 주제를 효과적으로 전달하는 역할을 합니다."},
    {"label": 0, "question": "강의에서 시적 화자와 서술자가 문학 공부에서 왜 중요한지 언제 설명해주나요?", "answer": "강의의 도입인 (00:51.360 - 00:59.360)에서 이야기하고 있습니다."},

    {"label": 1, "question": "소설이란 무엇인가요?", "answer": "소설은 작가가 상상력을 발휘하여 꾸며낸 허구적인 이야기로, 현실에서 일어날 법한 일을 기반으로 합니다."},
    {"label": 0, "question": "강의에서 소설의 개념 설명은 언제 나오나요?", "answer": "강의의 (01:00.000 - 01:14.460)에서 이야기하고 있습니다."},

    {"label": 1, "question": "소설의 주요 특징은 무엇인가요?", "answer": "소설은 작가가 꾸며낸 허구의 이야기이며, 인물과 사건의 전개를 통해 현실의 이야기인 것처럼 만들어낸 글입니다."},
    {"label": 0, "question": "소설의 주요 특징은 무엇인지 언제 설명하고 있나요?", "answer": "강의의 (01:14.880 - 01:29.940)에서 이야기하고 있습니다."},

    {"label": 1, "question": "서술자는 무엇을 하는 역할인가요?", "answer": "서술자는 작가가 이야기를 전달하기 위해 내세운 목소리로, 작품 속 상황을 바라보고 독자에게 전달하는 역할을 합니다."},
    {"label": 0, "question": "서술자는 무엇을 하는 역할인지 강의에서 언제 설명해주나요?", "answer": "강의의 (01:30.000 - 01:59.980)에서 이야기하고 있습니다."},

    {"label": 1, "question": "서술자와 작가는 같은 개념인가요?", "answer": "아니요, 서술자와 작가는 같은 개념이 아닙니다. 서술자는 작가가 내세운 목소리로, 작가와 분리된 존재입니다."},
    {"label": 0, "question": "서술자와 작가의 개념이 같은지 아닌지를 설명해주는 부분이 어디인가요?", "answer": "강의의 (01:59.980 - 02:05.020)에서 이야기하고 있습니다."},

    {"label": 1, "question": "서술자의 주요 특징은 무엇인가요?", "answer": "서술자는 작품 속 상황을 바라보고 독자에게 전달하며, 작품의 분위기를 형성하고 주제를 효과적으로 전달합니다."},
    {"label": 0, "question": "서술자의 주요 특징은 무엇인지 설명해주는 부분이 언제인가요?", "answer": "강의의 (02:06.680 - 02:11.320)에서 이야기하고 있습니다."},

    {"label": 1, "question": "서술자의 관점에 따라 무엇이 달라질 수 있나요?", "answer": "서술자의 관점에 따라 작품 속 대상에 대한 이해나 판단이 달라질 수 있습니다."},
    {"label": 0, "question": "서술자의 관점에 따라 무엇이 달라질 수 있는지 언제 설명되고 있나요?", "answer": "강의의 (02:11.820 - 02:15.000)에서 이야기하고 있습니다."},

    {"label": 1, "question": "서술자는 어디에서 이야기를 전달할 수 있나요?", "answer": "서술자는 작품 속 인물이 될 수도 있고, 작품 바깥에서 서술할 수도 있습니다."},
    {"label": 0, "question": "서술자는 어디에서 이야기를 전달할 수 있는지 설명되는 부분이 언제인가요?", "answer": "강의의 (02:21.040 - 02:22.420)에서 이야기하고 있습니다."},

    {"label": 1, "question": "서술자의 관점에 따라 소설은 어떻게 감상해야 하나요?", "answer": "서술자의 관점에 주목하여 소설을 감상하면, 작품의 깊이 있는 수용이 가능해집니다."},
    {"label": 0, "question": "서술자의 관점에 따라 소설은 어떻게 감상해야 하는지를 강의에서 몇분 즈음 설명해주나요?", "answer": "강의의 (02:29.240 - 02:35.640)에서 이야기하고 있습니다."},

    {"label": 1, "question": "서술자의 관점은 작품 속 세계에 어떤 영향을 미치나요?", "answer": "서술자의 관점에 따라 작품 속 세계가 다르게 형상화되고, 작품의 분위기와 주제가 다르게 형성됩니다."},
    {"label": 0, "question": "강의에서 언제 서술자의 관점은 작품 속 세계에 어떤 영향을 미치는지를 설명해주나요?", "answer": "강의의 (02:37.380 - 02:46.360)에서 이야기하고 있습니다."},

    {"label": 1, "question": "소설의 시점이란 무엇인가요?", "answer": "소설의 시점은 서술자가 이야기를 서술하는 위치나 사건을 바라보는 관점을 말합니다."},
    {"label": 0, "question": "소설의 시점이란 무엇인지는 강의에서 언제 설명되나요?", "answer": "강의의 (02:59.760 - 03:08.140)에서 이야기하고 있습니다."},

    {"label": 1, "question": "시점과 서술상의 특징은 어떤 관계가 있나요?", "answer": "시점과 서술상의 특징은 긴밀한 관계가 있으며, 작가가 내용을 어떻게 전달하고자 하는지에 대한 생각에 기반합니다."},
    {"label": 0, "question": "시점과 서술상의 특징은 어떤 관계가 있는지는 강의에서 몇분 쯤 설명해주나요?", "answer": "강의의 (03:13.920 - 03:29.220)에서 이야기하고 있습니다."},

    {"label": 1, "question": "시점과 서술상의 특징을 함께 공부해야 하는 이유는 무엇인가요?", "answer": "시점과 서술상의 특징은 함께 공부해야만 작품의 서술 구조를 온전히 이해할 수 있기 때문입니다."},
    {"label": 0, "question": "시점과 서술상의 특징을 함께 공부해야 하는 이유를 강의에 몇분 몇초에서 설명하고 있나요?", "answer": "강의 (03:34.460 - 03:39.080)에서 이야기하고 있습니다."}
]


df = pd.DataFrame(data)

# 일상적인 대화 데이터셋 추가 (라벨 2)
data2 = [
    {"label": 2, "question": "안녕하세요~", "answer": "안녕하세요. 오늘 공부도 화이팅하세요."},
    {"label": 2, "question": "오늘 날씨 어떤 거 같아", "answer": "네, 오늘 날씨는 문학을 공부하기 아주 좋은 날씨에요."},
    {"label": 2, "question": "공부하기 싫어", "answer": "그래도 공부하셔야 해요. 국어공부는 즐거운 일이에요~"},
]

df2 = pd.DataFrame(data2)

# KOBERT 모델 로드
tokenizer = BertTokenizer.from_pretrained("monologg/kobert")
model = BertModel.from_pretrained("monologg/kobert")

def get_dictionary_definition(word):
    certkey_no = '실제_인증키'  # 실제 인증키 입력
    API_KEY = '0BDDC23E6C674F8550A06A0D6EF4830E'  # 실제 API 키 입력
    url = f'https://opendict.korean.go.kr/api/search?certkey_no={certkey_no}&key={API_KEY}&type_search=search&q={word}&req_type=json'
    
    # API 호출
    response = requests.get(url)
    
    # 상태 코드 확인
    if response.status_code != 200:
        return [f"API 요청에 실패했습니다. 상태 코드: {response.status_code}"]

    # JSON 응답 파싱
    result = response.json()
    
    # 전체 JSON 응답 출력하여 구조 확인
    print(result)  # 응답 결과 확인
    
    # 정의를 추출하는 로직 수정
    if 'channel' in result and 'item' in result['channel']:
        # 'item' 리스트 안에 정의가 있는지 확인하고 정의를 반환
        definitions = []
        for entry in result['channel']['item'][:3]:  # 최대 3개의 정의 가져오기
            if 'sense' in entry and entry['sense']:
                definition = entry['sense'][0].get('definition', '정의 없음') + "\n"
                definitions.append(definition)
            else:
                definitions.append('정의 없음')
        return definitions if definitions else ["해당 단어의 정의를 찾을 수 없습니다."]
    else:
        return ["사전에서 항목을 찾을 수 없습니다."]

# 질문 임베딩 함수
def embed_questions(questions):
    inputs = tokenizer(questions, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :]

# 사전인지 일상적인 대화인지 확인하는 함수
def answer_question(user_question, df, df2):
    if user_question.startswith("사전)"):
        word = user_question.split("사전)")[1].strip()
        return "\n".join(get_dictionary_definition(word))
    
    question_embeddings = embed_questions(df['question'].tolist())
    user_embedding = embed_questions([user_question])
    similarities = cosine_similarity(user_embedding, question_embeddings)
    best_match_index = similarities.argmax()
    
    # 유사도가 0.5 이하일 경우 df2에서 일상 대화를 찾음
    if similarities[0][best_match_index] > 0.5:
        return df.iloc[best_match_index]['answer']
    else:
        # df2에서 일상적인 대화를 찾는 로직 추가
        question_embeddings_df2 = embed_questions(df2['question'].tolist())
        similarities_df2 = cosine_similarity(user_embedding, question_embeddings_df2)
        best_match_index_df2 = similarities_df2.argmax()

        if similarities_df2[0][best_match_index_df2] > 0.3:
            return df2.iloc[best_match_index_df2]['answer']
        else:
            return "답변을 찾을 수 없습니다."

# Streamlit 앱 시작
st.title("M-Buddy")
user_question = st.text_input("질문을 입력하세요:")

if user_question:
    answer = answer_question(user_question, df, df2)
    st.write(f"답변: {answer}")
