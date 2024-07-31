# to fast api
from fastapi import FastAPI, Request, Query
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

# to base
from dotenv import load_dotenv
import os, requests, json


# to data
from pykrx import stock
import pandas as pd

# .env 파일 로드
load_dotenv()

# 환경 변수 읽기
DART_API_KEY = os.getenv("DART_API_KEY")
CLOVA_HOST = os.getenv("CLOVA_HOST")
CLOVA_API_KEY = os.getenv("CLOVA_API_KEY")
CLOVA_API_KEY_PRIMARY_VAL = os.getenv("CLOVA_API_KEY_PRIMARY_VAL")
CLOVA_REQUEST_ID = os.getenv("REQUEST_ID")

app = FastAPI()
# Jinja2 템플릿 설정
templates = Jinja2Templates(directory="templates")

# static 디렉토리를 정적 파일 제공 경로로 설정
app.mount("/static", StaticFiles(directory="templates/static"), name="static")


industries = {
    4: "음식료품",
    5: "섬유의복",
    6: "종이목재",
    7: "화학",
    8: "의약품",
    9: "비금속광물",
    10: "철강금속",
    11: "기계",
    12: "전기전자",
    13: "의료정밀",
    14: "운수장비",
    15: "유통업",
    16: "전기가스업",
    17: "건설업",
    18: "운수창고업",
    19: "통신업",
    20: "금융업",
    21: "증권",
    22: "보험",
    23: "서비스업",
    24: "제조업",
}
now_date = "20240730"


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(
        "main.html", {"request": request, "industries": industries}
    )


#############################################################################################################################
########################################주식 데이터 가져오기###################################################################
#############################################################################################################################

# CSV 파일 읽기
index_info = pd.read_csv("index_info.csv", dtype=str)
corp_info = pd.read_csv("corp_info.csv", dtype=str)


##### 인덱스 PER, PBR 가져오기
def get_index_PERnPBR(index_name):
    # name 컬럼이 "건설업"인 행 필터링
    filtered_rows = index_info[index_info["name"] == index_name]
    result = filtered_rows["ticker"]
    result = stock.get_index_fundamental(now_date, now_date, str(result.values[0]))
    # print(result.head())

    return result.iloc[:, [2, 3]]


##### 인덱스 종목 가져오기
def get_corp_list_by_index(index_name):
    # name 컬럼이 "건설업"인 행 필터링
    filtered_rows = index_info[index_info["name"] == index_name]
    result = filtered_rows["corp_list"].values[0].split(",")

    return result


##### 종목 PER, PBR 가져오기
def get_corp_PERnPBR_by_index(index_name):
    # print(index_name)
    output = pd.DataFrame(columns=["corp_name", "PER", "PBR"])
    for stock_code in get_corp_list_by_index(index_name):
        # print(stock_code)
        # 필터링
        filtered_rows = corp_info[corp_info["stock_code"] == stock_code]
        corp_name = filtered_rows["corp_name"].values[0]
        result = stock.get_market_fundamental(now_date, now_date, stock_code)
        # print(result)
        # print(f"사명 : {corp_name}, PER : {result['PER'][0]}, PBR : {result['PBR'][0]}")

        if "PER" in result.columns and "PBR" in result.columns:
            new_row = pd.DataFrame(
                {
                    "corp_name": corp_name,
                    "PER": result["PER"].iloc[0],
                    "PBR": result["PBR"].iloc[0],
                },
                index=[len(output)],
            )
            output = pd.concat([output, new_row], ignore_index=True)
        # else: # PER, PBR 정보가 없는 경우는 버림
        # new_row = pd.DataFrame(
        #     {
        #         "corp_name": corp_name,
        #         "PER": "N/A",
        #         "PBR": "N/A",
        #     },
        #     index=[len(output)],
        # )
        # print(len(result), new_row)
    return output


#############################################################################################################################
########################################인덱스 페이지 작업###################################################################
#############################################################################################################################


@app.get("/{item_id}", response_class=HTMLResponse)
async def read_detail(request: Request, item_id: int):
    index_data = get_index_PERnPBR(industries[item_id])
    stocks = get_corp_list_by_index(industries[item_id])

    # get stock data
    index_stocks_info = get_corp_PERnPBR_by_index(industries[item_id])
    # print(index_stocks_info)

    index_PER = index_data["PER"].values[0]
    index_PBR = index_data["PBR"].values[0]
    index_mean_per = index_stocks_info["PER"].mean()
    index_mean_pbr = index_stocks_info["PBR"].mean()

    data = index_stocks_info.apply(
        lambda row: {"x": row["PER"], "y": row["PBR"], "id": row["corp_name"]}, axis=1
    ).tolist()
    return templates.TemplateResponse(
        "index_detail.html",
        {
            "request": request,
            "industries": industries,
            "industry_name": industries[item_id],
            "item_id": item_id,
            "stocks": stocks,
            "data": data,
            "index_PER": index_PER,
            "index_PBR": index_PBR,
            "index_mean_per": index_mean_per,
            "index_mean_pbr": index_mean_pbr,
        },
    )


def filter_and_transform(data):
    result = {}
    for item in data["list"]:
        if item.get("fs_nm") == "연결재무제표":
            account_nm = "당기누적 " + item.get("account_nm", "")
            thstrm_add_amount = item.get("thstrm_amount", "")
            result[account_nm] = thstrm_add_amount
    return result


#############################################################################################################################
########################################stock 페이지 작업###################################################################
#############################################################################################################################

##### 회사 재무 정보 가져오기 #####


def get_corpCode_from_corpName(corp_name):
    # CSV 파일 읽기
    df = pd.read_csv("corp_info.csv", dtype="str")
    # print(df)
    # stock_code로 필터링
    filtered_rows = df[df["corp_name"] == corp_name]
    return str(filtered_rows.iloc[0, 1])


def filter_and_transform(data):
    result = {}
    for item in data["list"]:
        if item.get("fs_nm") == "연결재무제표":
            account_nm = "당기누적 " + item.get("account_nm", "")
            thstrm_add_amount = item.get("thstrm_amount", "")
            result[account_nm] = thstrm_add_amount
    return result


def get_corp_FI(corp_name):
    corp_code = get_corpCode_from_corpName(corp_name)
    # API 키와 기본 URL 설정
    api_key = DART_API_KEY

    account_url = "https://opendart.fss.or.kr/api/fnlttSinglAcnt.json"
    # 요청 파라미터 설정
    params = {
        "crtfc_key": api_key,
        "corp_code": str(corp_code),
        "bsns_year": "2023",  # 사업 연도
        "reprt_code": "11011",  # 1분기보고서 : 11013, 반기보고서 : 11012, 3분기보고서 : 11014, 사업보고서 : 11011
        "idx_cl_code": "M210000",  # 수익성지표 : M210000 안정성지표 : M220000 성장성지표 : M230000 활동성지표 : M240000
    }

    # API 호출
    response = requests.get(account_url, params=params)

    # 응답 데이터 처리
    if response.status_code == 200:
        data = response.json()
        # print(json.dumps(data, indent=4, ensure_ascii=False))
        return filter_and_transform(data)
    else:
        return f"Error: {response.status_code}"


##### 회사 임원 정보 가져오기 #####


def extract_execute_from_dict(data):
    result = []
    for item in data["list"]:
        chrg_job = item.get("chrg_job", "")
        nm = item.get("nm", "")
        main_career = item.get("main_career", "")
        transformed_string = f"{chrg_job} - {nm} ({main_career})"
        result.append(transformed_string)
    return result


def get_corp_execs(corp_name):
    corp_code = get_corpCode_from_corpName(corp_name)
    # API 키와 기본 URL 설정
    api_key = DART_API_KEY

    url = "https://opendart.fss.or.kr/api/exctvSttus.json"

    # 요청 파라미터 설정
    params = {
        "crtfc_key": api_key,
        "corp_code": corp_code,  # 예시: 삼성전자
        "bsns_year": "2023",  # 사업 연도
        "reprt_code": "11011",  # 반기 보고서
    }

    # API 호출
    response = requests.get(url, params=params)

    # 응답 데이터 처리
    if response.status_code == 200:
        data = response.json()
        # JSON 데이터를 보기 좋게 출력
        # print(json.dumps(data, indent=4, ensure_ascii=False))
        return extract_execute_from_dict(data)
    else:
        return f"Error: {response.status_code}"


@app.get("/detail/", response_class=HTMLResponse)
async def detail_page(request: Request, index_id: int, corp_name: str):
    try:
        index_data = get_index_PERnPBR(industries[index_id])
        stocks = get_corp_list_by_index(industries[index_id])

        # get stock data
        index_stocks_info = get_corp_PERnPBR_by_index(industries[index_id])
        # print(index_stocks_info)

        index_PER = index_data["PER"].values[0]
        index_PBR = index_data["PBR"].values[0]
        index_mean_per = index_stocks_info["PER"].mean()
        index_mean_pbr = index_stocks_info["PBR"].mean()

        data = index_stocks_info.apply(
            lambda row: {"x": row["PER"], "y": row["PBR"], "id": row["corp_name"]},
            axis=1,
        ).tolist()

        corp_FI = get_corp_FI(corp_name)
        corp_execs = get_corp_execs(corp_name)

        return templates.TemplateResponse(
            "stock_detail.html",
            {
                "request": request,
                "industries": industries,
                "industry_name": industries[index_id],
                "item_id": index_id,
                "index_data": index_data,
                "stocks": stocks,
                "data": data,
                "index_PER": index_PER,
                "index_PBR": index_PBR,
                "index_mean_per": index_mean_per,
                "index_mean_pbr": index_mean_pbr,
                "corp_name": corp_name,
                # add financial data
                "current_period_total_assets": (
                    corp_FI["당기누적 자산총계"]
                    if "당기누적 자산총계" in corp_FI.keys()
                    else "N/A"
                ),
                "current_period_total_liabilities": (
                    corp_FI["당기누적 부채총계"]
                    if "당기누적 자산총계" in corp_FI.keys()
                    else "N/A"
                ),
                "current_period_total_sales": (
                    corp_FI["당기누적 매출액"]
                    if "당기누적 매출액" in corp_FI.keys()
                    else "N/A"
                ),
                "current_period_total_comprehensive_income": (
                    corp_FI["당기누적 총포괄손익"]
                    if "당기누적 총포괄손익" in corp_FI.keys()
                    else "N/A"
                ),
                # add executes data
                "execs": corp_execs,
            },
        )
    except Exception as e:
        return templates.TemplateResponse(
            "stock_detail.html",
            {
                "request": request,
                "industries": "[데이터 관련 에러 발생]",
            },
        )


#############################################################################################################################
########################################clova 페이지 작업###################################################################
#############################################################################################################################
def get_stockcode_from_corpName(corp_name):
    df = pd.read_csv("corp_info.csv", dtype="str")
    filtered_rows = df[df["corp_name"] == corp_name]
    # print(filtered_rows)
    return str(filtered_rows.iloc[0, 0])


def get_corp_stockNperNpbr(stock_code):
    row1 = stock.get_market_fundamental("20240730", "20240730", str(stock_code))
    row2 = stock.get_market_ohlcv("20240730", "20240730", "005930")
    new_row = {"시가": row2["시가"][0], "PER": row1["PER"][0], "PBR": row1["PBR"][0]}
    return new_row


# Clova 관련 클래스와 함수 추가
def bytes_to_dict(line):
    # 바이트 데이터를 문자열로 변환
    json_string = line.split("data:")[1].strip()
    json_dict = json.loads(json_string)
    return json_dict["message"]["content"]


class CompletionExecutor:
    def __init__(self, host, api_key, api_key_primary_val, request_id):
        self._host = host
        self._api_key = api_key
        self._api_key_primary_val = api_key_primary_val
        self._request_id = request_id

    def execute(self, completion_request):
        headers = {
            "X-NCP-CLOVASTUDIO-API-KEY": self._api_key,
            "X-NCP-APIGW-API-KEY": self._api_key_primary_val,
            "X-NCP-CLOVASTUDIO-REQUEST-ID": self._request_id,
            "Content-Type": "application/json; charset=utf-8",
            "Accept": "text/event-stream",
        }

        result = []
        with requests.post(
            self._host + "/testapp/v1/chat-completions/HCX-003",
            headers=headers,
            json=completion_request,
            stream=True,
        ) as r:
            tmp = []
            for line in r.iter_lines():
                if line:
                    line = line.decode("utf-8")
                    if line.startswith("data:") and "message" in line:
                        line = bytes_to_dict(line)
                        if line.startswith("\n"):
                            if tmp:
                                result.append("".join(tmp))
                                tmp = []
                        else:
                            tmp.append(line)
            result.append("".join(tmp[0:-1]))
        # 결과를 문자열로 반환
        return result


def analyze_company(index, corp_name):
    # print("start analyzing")
    completion_executor = CompletionExecutor(
        host=CLOVA_HOST,
        api_key=CLOVA_API_KEY,
        api_key_primary_val=CLOVA_API_KEY_PRIMARY_VAL,
        request_id=CLOVA_REQUEST_ID,
    )

    # get account info
    stock_code = get_stockcode_from_corpName(corp_name)
    # print("stock code: ", stock_code)
    account = get_corp_FI(corp_name)
    # print("account: ", account)
    # get board members
    stocks = get_corp_stockNperNpbr(stock_code)
    # print("stocks: ", stocks)
    board_members = get_corp_execs(corp_name)
    # print("board members: ", board_members)

    query = f"이 기업의 이름은 {corp_name}이며 산업군은 {index}입니다.\n\n\n재무상태는 다음과 같습니다.\n{stocks}\n{account}  \n\n\n경영진 정보는 다음과 같습니다.\n이사회: {board_members}\n\n\n이 데이터를 분석하여 이 회사가 좋은 투자 기회인지 판단해 주세요. 추천 이유를 상세히 설명해 주세요.\n 다음 양식에 따라 답변해주세요. \n 투자 여부 판단 : 좋음 / 보통 / 나쁨 \n 이유 : "
    preset_text = [
        {
            "role": "system",
            "content": "당신은 금융 분석 및 투자 조언에 특화된 AI 어시스턴트입니다. 회사의 산업 분류, 재무 상태, 경영진 정보가 제공될 것입니다. 이 데이터를 기반으로 주식이 좋은 투자 기회인지 여부를 판단하십시오. 상세한 분석과 추천 이유를 제공하세요.",
        },
        {"role": "user", "content": query},
    ]
    request_data = {
        "messages": preset_text,
        "topP": 0.8,
        "topK": 0,
        "maxTokens": 256,
        "temperature": 0.5,
        "repeatPenalty": 5.0,
        "stopBefore": [],
        "includeAiFilters": True,
        "seed": 0,
    }

    return completion_executor.execute(request_data)


@app.get("/clova/", response_class=HTMLResponse)
async def detail_page(request: Request, index_id: int, corp_name: str):
    index_data = get_index_PERnPBR(industries[index_id])
    # get stock data
    index_stocks_info = get_corp_PERnPBR_by_index(industries[index_id])
    # print(index_stocks_info)
    stock_code = get_stockcode_from_corpName(corp_name)
    result = stock.get_market_fundamental(now_date, now_date, stock_code)

    index_PER = index_data["PER"].values[0]
    index_PBR = index_data["PBR"].values[0]
    index_mean_per = index_stocks_info["PER"].mean()
    index_mean_pbr = index_stocks_info["PBR"].mean()

    data = index_stocks_info.apply(
        lambda row: {"x": row["PER"], "y": row["PBR"], "id": row["corp_name"]}, axis=1
    ).tolist()

    analysis_result = analyze_company(industries[index_id], corp_name)
    # print(analysis_result)
    return templates.TemplateResponse(
        "clova_detail.html",
        {
            "request": request,
            "industries": industries,
            "industry_name": industries[index_id],
            "item_id": index_id,
            "index_data": index_data,
            "data": data,
            "index_PER": index_PER,
            "index_PBR": index_PBR,
            "index_mean_per": index_mean_per,
            "index_mean_pbr": index_mean_pbr,
            "corp_name": corp_name,
            "analysis_result": analysis_result,
        },
    )
