import requests
import streamlit as st
import pandas as pd
from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

# 기존 질문-답변 데이터
data = [
    {"label": 1, "question": "시의 정의", "answer": "마음속에 떠오르는 생각나 느낌을 운율 있는 언어로 압축하여 함축적으로 나타낸 문학"},
    {"label": 1, "question": "시의 구성 요소", "answer": "의미적 요소: 시인이 전달하려는 사상이나 생각(주제), 회화적 요소: 시를 읽을 때 마음속에 떠오르는 감각적인 모습이나 느낌(심상), 음악적 요소: 시에서 일정하게 반복적으로 나타나는 소리의 규칙적인 가락(운율)"},
    {"label": 1, "question": "시적 화자의 정의", "answer": "서정적 자아, 시적 자아, 말하는 이. 시에서 말하는 이로, 시인이 자신의 생각과 느낌을 효과적으로 전달하기 위해 내세운 인물"},
    {"label": 1, "question": "시적 화자의 특징", "answer": "시인을 대신해 시적 정서와 주제를 효과적으로 전달함. 대상에 대한 시인의 생각이나 태도, 독특한 말투가 드러남. 시 속에 직접 드러나기도 하고 드러나지 않기도 함. 어떤 화자를 선택하는지에 따라 작품의 내용과 분위기 등이 달라짐"},
    {"label": 1, "question": "시적 화자의 위치", "answer": "겉으로 드러나는 경우 (표면적 화자), '나', '내(가)', '우리'라는 표현이 나타남"},
    {"label": 1, "question": "시적 화자의 위치", "answer": "겉으로 드러나지 않는 경우 (이면적 화자), '나', '내(가)', '우리'라는 표현이 나타나지 않음"},
    {"label": 1, "question": "서시의 내용", "answer": "서시: 죽는 날까지 하늘을 우러러 한 점 부끄럼이 없기를, 잎새에 이는 바람에도 나는 괴로워했다"},
    {"label": 1, "question": "서시에서의 시적 화자", "answer": "자신의 삶을 성찰하는 '나'"},
    {"label": 1, "question": "엄마 걱정", "answer": "엄마 걱정: 열무 삼십 단을 이고 시장에 간 우리 엄마 안 오시네, 해는 시든 지 오래 나는 찬밥처럼 방에 담겨 아무리 천천히 숙제를 해도 엄마 안 오시네, 배춧잎 같은 발소리 타박타박 암 들리네, 어둡고 무서워"},
    {"label": 1, "question": "엄마 걱정에서의 시적 화자 위치", "answer": "작품 속, 겉으로 드러나는 경우"},
    {"label": 1, "question": "나그네 내용", "answer": "나그네: 강나루 건너서 밀밭 길을 구름에 달 가듯이 가는 나그네, 길은 외줄기, 남도 삼백 리, 술 익는 마을마다 타는 저녁놀, 구름에 달 가듯이 가는 나그네"},
    {"label": 1, "question": "나그네에서의 시적 화자", "answer": "시적 화자가 시의 표면에 (겉에) 드러나지 않았다"},
    {"label": 1, "question": "시적 화자를 중심으로 시 감상하기, 시적 화자와 관련된 내용을 파악하는 방법", "answer": "1) 시를 읽으며 시적 화자가 어디에 있으며 어떤 인물인지를 파악한다. 2) 화자가 누구인지 찾고 화자가 자신의 이야기를 하고 있는지, 관찰자의 위치에 있는지 확인한다. 3) 시적 화자가 처한 상황이 어떠한지 파악한다 4) 시적 화자가 상황에 대해서 어떤 반응이나 태도를 보이는지 생각한다 5) 시적 화자의 감정 변화 등에 초점을 맞추어 화자의 정서를 추리한다 6) 시적 화자의 행동, 의지 표현 등을 통해 화자의 태도를 추리하고, 화자의 새로운 깨달음, 생각 등을 통해 화자의 인식을 파악한다 7) 화자의 반응, 태도를 통해 시인이 말하고자 하는 것이 무엇인지 생각한다"},
    {"label": 1, "question": "시적 화자의 태도의 개념", "answer": "시에 나타난 상황이나 대상에 대한 시적 화자의 심리적 자세나 대응 방식"},
    {"label": 1, "question": "시적 화자의 긍정적 태도의 개념", "answer": "현실이나 미래의 상황이 잘 풀릴 것이라고 전망함. 상황이나 대상이 옳다고 인정하거나 바람직하다고 받아들임"},
    {"label": 1, "question": "시적 화자의 긍정적 태도의 예시", "answer": "겨울은 바다와 대륙 밖에서 그 매운 눈보라 몰고 왔지만, 이제 올 너그러운 봄은 삼천리 마을마다 우리들 가슴속에서 움트리라. <봄은>, 신동엽"},
    {"label": 1, "question": "시적 화자의 부정적, 비관적 태도의 개념", "answer": "현실이나 미래 상황을 부정적으로 보며, 변화시킬 수 없다고 생각함. 부정적인 상황에 슬퍼하거나, 절망하거나, 체념하는 모습을 보임"},
    {"label": 1, "question": "시적 화자의 부정적 태도의 예시", "answer": "먼 들길을 애기가 간다. 맨발 벗은 애기가 울면서 간다. 불러도 대답이 없다. 그림자마저 아른거린다. <은수저>, 김광균"},
    {"label": 1, "question": "시적 화자의 비판적, 냉소적 태도의 개념", "answer": "상황을 부정적으로 느꼈을 때 나타나는 태도. 대상의 잘못을 지적하여 옳고 그름을 가리는 태도. 대상을 부정적으로 바라보며 비웃는 태도"},
    {"label": 1, "question": "시적 화자의 비판적, 냉소적 태도의 예시", "answer": "성북동 산에 번지가 새로 생기면서 본래 살던 성북동 비둘기만이 번지가 없어졌다. 새벽부터 돌 깨는 산울림에 떨다가 가슴에 금이 갔다. 그래도 성북동 비둘기는 하느님의 광장 같은 새파란 아침 하늘에 성북동 주민에게 축복의 메시지나 전하듯 성북동 하늘을 한 바퀴 휘 돈다. <성북동 비둘기>, 김광섭"},
    {"label": 1, "question": "시적 화자의 자연 친화적 태도의 개념", "answer": "자연 속의 삶에 대한 지향이나 만족감이 드러나는 태도. 자연과 하나 되어 그것에 의지하거나, 그것과 하나가 되는 모습을 보임"},
    {"label": 1, "question": "시적 화자의 자연 친화적 태도의 예시", "answer": "남으로 창을 내겠소. 밭이 한참이 이로 파고 호미론 김을 매지요. <남으로 창을 내겠소>, 김상용"},
    {"label": 1, "question": "시적 화자의 동경적, 예찬적 태도의 개념", "answer": "대상의 장점을 높이 평가하여 그것을 그리워하거나, 찬양하고 기리는 태도. 자신의 처지나 현실에 만족하지 못할 때 드러날 수 있음"},
    {"label": 1, "question": "시적 화자의 동경적, 예찬적 태도의 예시", "answer": "엄마야 누나야 강변 살지. 뜰에는 반짝이는 금모래빛, 뒷문 밖에는 갈잎의 노래. 엄마야 누나야 강변 살자. <엄마야 누나야>, 김소월"},
    {"label": 1, "question": "시적 화자의 반성적 태도의 개념", "answer": "자신의 잘못을 되돌아보며 뉘우치는 태도. 과거나 현재의 삶에 일정한 거리를 두고 자신이 한 일을 되짚어 생각함"},
    {"label": 1, "question": "시적 화자의 반성적 태도의 예시", "answer": "창밖에 밤비가 속살거려 육첩방은 남의 나라, 시인이란 슬픈 천명인 줄 알면서도 한 줄 시를 적어 볼까. 땀내와 사랑 내 포근히 품긴 보내 주신 학비 봉투를 받아 대학 노-트를 끼고 늙은 교수의 강의 들으러 간다. 생각해 보면 어린 때 동무들 하나, 둘, 죄다 잃어버리고 나는 무얼 바라 나는 다만, 홀로 침전하는 것일까? 인생은 살기 어렵다는데 시가 이렇게 쉽게 씌어지는 것은 부끄러운 일이다. <쉽게 씨어진 시>, 윤동주"},

    {"label": 1, "question": "시적 화자와 서술자", "answer": "시적 화자와 서술자는 문학에서 중요한 개념입니다. 서술자는 소설에서 이야기를 전달하는 목소리로, 작품의 주제를 효과적으로 전달하는 역할을 합니다."},
    {"label": 0, "question": "강의에서 시적 화자와 서술자가 문학 공부에서 왜 중요한지 언제", "answer": "강의의 도입인 (00:51.360 - 00:59.360)에서 이야기하고 있습니다."},

    {"label": 1, "question": "소설의 정의", "answer": "소설은 작가가 상상력을 발휘하여 꾸며낸 허구적인 이야기로, 현실에서 일어날 법한 일을 기반으로 합니다."},
    {"label": 0, "question": "강의에서 소설의 개념 강의에서 언제", "answer": "강의의 (01:00.000 - 01:14.460)에서 이야기하고 있습니다."},

    {"label": 1, "question": "소설의 주요 특징", "answer": "소설은 작가가 꾸며낸 허구의 이야기이며, 인물과 사건의 전개를 통해 현실의 이야기인 것처럼 만들어낸 글입니다."},
    {"label": 0, "question": "소설의 주요 특징은 무엇인지 강의에서 언제", "answer": "강의의 (01:14.880 - 01:29.940)에서 이야기하고 있습니다."},

    {"label": 1, "question": "서술자는 무엇을 하는 역할", "answer": "서술자는 작가가 이야기를 전달하기 위해 내세운 목소리로, 작품 속 상황을 바라보고 독자에게 전달하는 역할을 합니다."},
    {"label": 0, "question": "서술자는 무엇을 하는 역할인지 강의에서 언제", "answer": "강의의 (01:30.000 - 01:59.980)에서 이야기하고 있습니다."},

    {"label": 1, "question": "서술자와 작가는 같은 개념인가요", "answer": "아니요, 서술자와 작가는 같은 개념이 아닙니다. 서술자는 작가가 내세운 목소리로, 작가와 분리된 존재입니다."},
    {"label": 0, "question": "서술자와 작가의 개념이 같은지 아닌지 강의에서 언제", "answer": "강의의 (01:59.980 - 02:05.020)에서 이야기하고 있습니다."},

    {"label": 1, "question": "서술자의 주요 특징", "answer": "서술자는 작품 속 상황을 바라보고 독자에게 전달하며, 작품의 분위기를 형성하고 주제를 효과적으로 전달합니다."},
    {"label": 0, "question": "서술자의 주요 특징 언제", "answer": "강의의 (02:06.680 - 02:11.320)에서 이야기하고 있습니다."},

    {"label": 1, "question": "서술자의 관점에 따라 무엇이 달라질 수 있나요", "answer": "서술자의 관점에 따라 작품 속 대상에 대한 이해나 판단이 달라질 수 있습니다."},
    {"label": 0, "question": "서술자의 관점에 따라 무엇이 달라질 수 있는지 강의에서 언제", "answer": "강의의 (02:11.820 - 02:15.000)에서 이야기하고 있습니다."},

    {"label": 1, "question": "서술자는 어디에서 이야기를 전달", "answer": "서술자는 작품 속 인물이 될 수도 있고, 작품 바깥에서 서술할 수도 있습니다."},
    {"label": 0, "question": "서술자는 어디에서 이야기를 전달 강의에서 언제", "answer": "강의의 (02:21.040 - 02:22.420)에서 이야기하고 있습니다."},

    {"label": 1, "question": "서술자의 관점에 따라 소설은 어떻게 감상해야 하나요", "answer": "서술자의 관점에 주목하여 소설을 감상하면, 작품의 깊이 있는 수용이 가능해집니다."},
    {"label": 0, "question": "서술자의 관점에 따라 소설은 어떻게 감상해야 하는지를 강의에서 언제", "answer": "강의의 (02:29.240 - 02:35.640)에서 이야기하고 있습니다."},

    {"label": 1, "question": "서술자의 관점은 작품 속 세계에 어떤 영향", "answer": "서술자의 관점에 따라 작품 속 세계가 다르게 형상화되고, 작품의 분위기와 주제가 다르게 형성됩니다."},
    {"label": 0, "question": "서술자의 관점은 작품 속 세계에 어떤 영향을 미치는지 강의에서 언제", "answer": "강의의 (02:37.380 - 02:46.360)에서 이야기하고 있습니다."},

    {"label": 1, "question": "소설의 시점이", "answer": "소설의 시점은 서술자가 이야기를 서술하는 위치나 사건을 바라보는 관점을 말합니다."},
    {"label": 0, "question": "소설의 시점이란 무엇인지는 강의에서 언제", "answer": "강의의 (02:59.760 - 03:08.140)에서 이야기하고 있습니다."},

    {"label": 1, "question": "시점과 서술상의 특징은 어떤 관계", "answer": "시점과 서술상의 특징은 긴밀한 관계가 있으며, 작가가 내용을 어떻게 전달하고자 하는지에 대한 생각에 기반합니다."},
    {"label": 0, "question": "시점과 서술상의 특징은 어떤 관계가 있는지는 강의에서 언제", "answer": "강의의 (03:13.920 - 03:29.220)에서 이야기하고 있습니다."},

    {"label": 1, "question": "시점과 서술상의 특징을 함께 공부해야 하는 이유", "answer": "시점과 서술상의 특징은 함께 공부해야만 작품의 서술 구조를 온전히 이해할 수 있기 때문입니다."},
    {"label": 0, "question": "시점과 서술상의 특징을 함께 공부해야 하는 이유 언제", "answer": "강의 (03:34.460 - 03:39.080)에서 이야기하고 있습니다."}
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

        if similarities_df2[0][best_match_index_df2] > 0.5:
            return df2.iloc[best_match_index_df2]['answer']
        else:
            return "답변을 찾을 수 없습니다."

# Streamlit 앱 시작
st.title("M-Buddy")
user_question = st.text_input("질문을 입력하세요:")

if user_question:
    answer = answer_question(user_question, df, df2)
    st.write(f"답변: {answer}")
