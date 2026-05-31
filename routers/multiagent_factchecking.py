__author__ = "Dong Yihan"

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from pprint import pprint
import traceback
from dependencies import post_process
from fact_check_agents import (google_search_fact_check_agent, wikipedia_fact_check_agent, news_fact_check_agent,
                               language_model_agent)

router = APIRouter(
    prefix="/multi_agent",
    tags=["multi_agent_system"]
)

ERROR_STATUS = {
    "status": 500,
    "message": "error in multi_agent_fact_checking()"
}


class MultiAgentFactCheckJson(BaseModel):
    raw: str


class SimpleVotingBinaryFactCheckJson(BaseModel):
    claims_and_results_google: list
    claims_and_results_wiki: list
    claims_and_results_news: list


def check_status(status: dict):
    """
    エラーコードを検出する関数
    :param status: dict{"status", "message"}
    :return:
    """
    if status["status"] == 500:
        return False
    elif status["status"] == 200:
        return True


def convert_response_from_single_agent(agent_response_factuality: bool):
    """
    エージェントからの返事をbooleanからstrに変換する関数
    :param agent_response_factuality: boolean
    :return:
    """
    agent_response_factuality_str = ""
    if agent_response_factuality is True:
        agent_response_factuality_str = "TRUE"
    else:
        agent_response_factuality_str = "FALSE"
    return agent_response_factuality_str


# マルチエージェントファクトチェック
@router.post("/fact_checking")
async def multi_agent_fact_checking(multi_agent_fact_check_payload: MultiAgentFactCheckJson):
    """
    マルチエージェント事実検証における議論エージェントの開発はまだ決めていないですが、それに想定した上で、マルチエージェントの仕組みに
    基づくサービスがfastapiを用いて実装しました。

    :param multi_agent_fact_check_payload:
    :return:
    """
    global ERROR_STATUS
    try:
        # あるところから元の投稿を転送された
        user_post = multi_agent_fact_check_payload.raw

        # 投稿処理クラスの初期化
        post_process_functions = post_process.PostProcess()
        status = post_process_functions.status
        if not check_status(status=status):
            ERROR_STATUS = status
            return JSONResponse(content=status["message"], status_code=status["status"])

        # 各事実検証エージェントの初期化
        google_search_agent = google_search_fact_check_agent.GoogleSearchFactCheckAgent()
        status = google_search_agent.status
        if not check_status(status=status):
            ERROR_STATUS = status
            return JSONResponse(content=status["message"], status_code=status["status"])

        wikipedia_agent = wikipedia_fact_check_agent.WikipediaFactCheckAgent()
        status = wikipedia_agent.status
        if not check_status(status=status):
            ERROR_STATUS = status
            return JSONResponse(content=status["message"], status_code=status["status"])

        # news_agent = news_fact_check_agent.NewsFactCheckAgent()
        # status = news_agent.status
        # if not check_status(status):
        #     ERROR_STATUS = status
        #     return JSONResponse(content=status["message"], status_code=status["status"])
        llm_agent = language_model_agent.LanguageModelAgent()
        status = llm_agent.status
        if not check_status(status):
            ERROR_STATUS = status
            return JSONResponse(content=status["message"], status_code=status["status"])

        # 投稿から主張を抽出する・実験を行うときには下のメソッドを使って、そのまま主張を特定の形に変形する
        # claims = post_process_functions.claim_extraction(post=user_post)
        claims = post_process_functions.convert_post_to_claim(post=user_post)
        status = post_process_functions.status
        if not check_status(status=status):
            ERROR_STATUS = status
            return JSONResponse(content=status["message"], status_code=status["status"])

        # 主張を問題の形に変換する
        queries = post_process_functions.query_generation(claim_list=claims)
        status = post_process_functions.status
        if not check_status(status=status):
            ERROR_STATUS = status
            return JSONResponse(content=status["message"], status_code=status["status"])

        # fact checking agentの起動。主張、各エージェントの判断結果およびそれぞれの確信度を返信する
        claims_and_results_google = google_search_agent.google_search_agent_workflow(query_list=queries,
                                                                                     claim_list=claims)
        status = google_search_agent.status
        if not check_status(status=status):
            ERROR_STATUS = status
            return JSONResponse(content=status["message"], status_code=status["status"])

        claims_and_results_wiki = wikipedia_agent.wikipedia_agent_workflow(query_list=queries,
                                                                           claim_list=claims)
        status = wikipedia_agent.status
        if not check_status(status=status):
            ERROR_STATUS = status
            return JSONResponse(content=status["message"], status_code=status["status"])

        # claims_and_results_news = news_agent.news_agent_workflow(claim_list=claims)
        # status = news_agent.status
        # if not check_status(status):
        #     ERROR_STATUS = status
        #     return JSONResponse(content=status["message"], status_code=status["status"])
        claims_and_results_llm = llm_agent.llm_agent_workflow(claim_list=claims, query_list=queries)
        status = llm_agent.status
        if not check_status(status):
            ERROR_STATUS = status
            return JSONResponse(content=status["message"], status_code=status["status"])

        # 各エージェントからもらった結果の確信度を計算する
        agent_results_list = [claims_and_results_google, claims_and_results_wiki, claims_and_results_llm]
        claims_and_sub_confidence = post_process_functions.calculate_sub_confidence(
            agent_results_list=agent_results_list)

        # 主張の確信度を合わせて、投稿の信憑性（しんぴょうせい）を計算する
        post_and_confidence = post_process_functions.judge_original_post_factuality(
            claims_and_sub_confidence=claims_and_sub_confidence, original_post=user_post
        )
        print("MAFC Score:", post_and_confidence["confidence"])

        return JSONResponse(content=post_and_confidence["factuality"], status_code=200)
    except Exception as err:

        traceback.print_exc()

        raise HTTPException(status_code=ERROR_STATUS["status"], detail=ERROR_STATUS["message"])


@router.post("/simple_voting_binary_fact_checking")
async def simple_voting_binary_fact_checking(simple_voting_fact_checking_payload: MultiAgentFactCheckJson):
    """
    MAFC Scoreを使わずに、単純に投票で事実検証の結果を獲得するメソッド
    :param simple_voting_fact_checking_payload:
    :return:
    """
    global ERROR_STATUS
    try:
        # 転送された内容の読み込み
        user_post = simple_voting_fact_checking_payload.raw

        # 投稿処理クラスの初期化
        post_process_functions = post_process.PostProcess()
        status = post_process_functions.status
        if not check_status(status):
            ERROR_STATUS = status
            return JSONResponse(content=ERROR_STATUS["message"], status_code=ERROR_STATUS["status"])

        # 各エージェントの初期化
        google_search_agent = google_search_fact_check_agent.GoogleSearchFactCheckAgent()
        status = google_search_agent.status
        if not check_status(status):
            ERROR_STATUS = status
            return JSONResponse(content=ERROR_STATUS["message"], status_code=ERROR_STATUS["status"])

        wikipedia_agent = wikipedia_fact_check_agent.WikipediaFactCheckAgent()
        status = wikipedia_agent.status
        if not check_status(status):
            ERROR_STATUS = status
            return JSONResponse(content=ERROR_STATUS["message"], status_code=ERROR_STATUS["status"])

        # news_agent = news_fact_check_agent.NewsFactCheckAgent()
        # status = news_agent.status
        # if not check_status(status):
        #     ERROR_STATUS = status
        #     return JSONResponse(content=ERROR_STATUS["message"], status_code=ERROR_STATUS["status"])
        llm_agent = language_model_agent.LanguageModelAgent()
        status = llm_agent.status
        if not check_status(status):
            ERROR_STATUS = status
            return JSONResponse(content=status["message"], status_code=status["status"])

        # 投稿から主張を抽出する・実験を行うときに下のメソッドを使って、そのまま実験データを主張リストの形に変換する
        # claims = post_process_functions.claim_extraction(post=user_post)
        claims = post_process_functions.convert_post_to_claim(post=user_post)
        status = post_process_functions.status
        if not check_status(status):
            ERROR_STATUS = status
            return JSONResponse(content=ERROR_STATUS["message"], status_code=ERROR_STATUS["status"])

        # 主張を質問の形に変形する
        queries = post_process_functions.query_generation(claim_list=claims)
        status = post_process_functions.status
        if not check_status(status):
            ERROR_STATUS = status
            return JSONResponse(content=ERROR_STATUS["message"], status_code=ERROR_STATUS["status"])

        # 各事実検証エージェントの起動。各エージェントが自らの判断結果を返す。
        claims_and_results_google = google_search_agent.google_search_agent_workflow(query_list=queries,
                                                                                     claim_list=claims)
        status = google_search_agent.status
        if not check_status(status):
            ERROR_STATUS = status
            return JSONResponse(content=ERROR_STATUS["message"], status_code=ERROR_STATUS["status"])

        claims_and_results_wiki = wikipedia_agent.wikipedia_agent_workflow(query_list=queries,
                                                                           claim_list=claims)
        status = wikipedia_agent.status
        if not check_status(status):
            ERROR_STATUS = status
            return JSONResponse(content=ERROR_STATUS["message"], status_code=ERROR_STATUS["status"])

        # claims_and_results_news = news_agent.news_agent_workflow(claim_list=claims)
        # status = news_agent.status
        # if not check_status(status):
        #     ERROR_STATUS = status
        #     return JSONResponse(content=ERROR_STATUS["message"], status_code=ERROR_STATUS["status"])
        claims_and_results_llm = llm_agent.llm_agent_workflow(claim_list=claims, query_list=queries)
        status = llm_agent.status
        if not check_status(status):
            ERROR_STATUS = status
            return JSONResponse(content=status["message"], status_code=status["status"])

        # 投票で元の投稿の信憑性を決める
        claims_and_results_list = [claims_and_results_google, claims_and_results_wiki, claims_and_results_llm]
        post_and_factuality = post_process_functions.judge_original_post_factuality_voting_binary(
            original_post=user_post, claims_and_results_list=claims_and_results_list
        )
        status = post_process_functions.status
        if not check_status(status):
            ERROR_STATUS = status
            return JSONResponse(content=ERROR_STATUS["message"], status_code=ERROR_STATUS["status"])

        return JSONResponse(content=post_and_factuality["factuality"], status_code=200)
    except Exception as err:

        traceback.print_exc()

        raise HTTPException(status_code=ERROR_STATUS["status"], detail=ERROR_STATUS["message"])


@router.post("/simple_voting_fact_checking")
async def simple_voting_multilabel_fact_checking(
        simple_voting_multilabel_fact_checking_payload: MultiAgentFactCheckJson):
    """
    MAFC Scoreを使わずに、単純に投票で事実検証の結果を獲得するメソッド
    :param simple_voting_multilabel_fact_checking_payload:
    :return:
    """
    global ERROR_STATUS
    try:
        # 転送された内容の読み込み
        user_post = simple_voting_multilabel_fact_checking_payload.raw

        # 投稿処理クラスの初期化
        post_process_functions = post_process.PostProcess()
        status = post_process_functions.status
        if not check_status(status):
            ERROR_STATUS = status
            return JSONResponse(content=ERROR_STATUS["message"], status_code=ERROR_STATUS["status"])

        # 各エージェントの初期化
        google_search_agent = google_search_fact_check_agent.GoogleSearchFactCheckAgent()
        status = google_search_agent.status
        if not check_status(status):
            ERROR_STATUS = status
            return JSONResponse(content=ERROR_STATUS["message"], status_code=ERROR_STATUS["status"])

        wikipedia_agent = wikipedia_fact_check_agent.WikipediaFactCheckAgent()
        status = wikipedia_agent.status
        if not check_status(status):
            ERROR_STATUS = status
            return JSONResponse(content=ERROR_STATUS["message"], status_code=ERROR_STATUS["status"])

        # news_agent = news_fact_check_agent.NewsFactCheckAgent()
        # status = news_agent.status
        # if not check_status(status):
        #     ERROR_STATUS = status
        #     return JSONResponse(content=ERROR_STATUS["message"], status_code=ERROR_STATUS["status"])
        llm_agent = language_model_agent.LanguageModelAgent()
        status = llm_agent.status
        if not check_status(status):
            ERROR_STATUS = status
            return JSONResponse(content=status["message"], status_code=status["status"])

        # 投稿から主張を抽出する・実験を行うときに下のメソッドを使って、そのまま実験データを主張リストの形に変換する
        # claims = post_process_functions.claim_extraction(post=user_post)
        claims = post_process_functions.convert_post_to_claim(post=user_post)
        status = post_process_functions.status
        if not check_status(status):
            ERROR_STATUS = status
            return JSONResponse(content=ERROR_STATUS["message"], status_code=ERROR_STATUS["status"])

        # 主張を質問の形に変形する
        queries = post_process_functions.query_generation(claim_list=claims)
        status = post_process_functions.status
        if not check_status(status):
            ERROR_STATUS = status
            return JSONResponse(content=ERROR_STATUS["message"], status_code=ERROR_STATUS["status"])

        # 各事実検証エージェントの起動。各エージェントが自らの判断結果を返す。
        claims_and_results_google = google_search_agent.google_search_agent_workflow(query_list=queries,
                                                                                     claim_list=claims)
        status = google_search_agent.status
        if not check_status(status):
            ERROR_STATUS = status
            return JSONResponse(content=ERROR_STATUS["message"], status_code=ERROR_STATUS["status"])

        claims_and_results_wiki = wikipedia_agent.wikipedia_agent_workflow(query_list=queries,
                                                                           claim_list=claims)
        status = wikipedia_agent.status
        if not check_status(status):
            ERROR_STATUS = status
            return JSONResponse(content=ERROR_STATUS["message"], status_code=ERROR_STATUS["status"])

        # claims_and_results_news = news_agent.news_agent_workflow(claim_list=claims)
        # status = news_agent.status
        # if not check_status(status):
        #     ERROR_STATUS = status
        #     return JSONResponse(content=ERROR_STATUS["message"], status_code=ERROR_STATUS["status"])
        claims_and_results_llm = llm_agent.llm_agent_workflow(claim_list=claims, query_list=queries)
        status = llm_agent.status
        if not check_status(status):
            ERROR_STATUS = status
            return JSONResponse(content=status["message"], status_code=status["status"])

        # 投票で元の投稿の信憑性を決める
        claims_and_results_list = [claims_and_results_google, claims_and_results_wiki, claims_and_results_llm]
        post_and_factuality = post_process_functions.judge_original_post_factuality_voting_multilabel(
            original_post=user_post, claims_and_results_list=claims_and_results_list
        )
        status = post_process_functions.status
        if not check_status(status):
            ERROR_STATUS = status
            return JSONResponse(content=ERROR_STATUS["message"], status_code=ERROR_STATUS["status"])

        return JSONResponse(content=post_and_factuality["factuality"], status_code=200)
    except Exception as err:

        traceback.print_exc()

        raise HTTPException(status_code=ERROR_STATUS["status"], detail=ERROR_STATUS["message"])


@router.post("/single_agent_fact_checking")
async def single_agent_fact_checking(single_agent_fact_checking_payload: MultiAgentFactCheckJson):
    """
    複数のエージェントで行った実験と比べるため、一つのエージェントで事実検証を行うメソッドを作ります。

    一つのエージェントの場合には、MAFC Scoreの計算は不要となります。
    :param single_agent_fact_checking_payload:
    :return:
    """
    global ERROR_STATUS
    try:
        # 転送された投稿内容の読み込み
        user_post = single_agent_fact_checking_payload.raw

        # 投稿処理クラスの初期化
        post_process_functions = post_process.PostProcess()
        status = post_process_functions.status
        if not check_status(status):
            ERROR_STATUS = status
            return JSONResponse(content=ERROR_STATUS["message"], status_code=ERROR_STATUS["status"])

        # 一つのエージェントの初期化。Googleエージェントでの実験もう終わりましたので、次にはWikipediaエージェントです。
        news_agent = news_fact_check_agent.NewsFactCheckAgent()
        status = news_agent.status
        if not check_status(status):
            ERROR_STATUS = status
            return JSONResponse(content=ERROR_STATUS["message"], status_code=ERROR_STATUS["status"])

        # 主張を投稿から抽出する・実験を行うときには下のメソッドを使って、そのまま投稿を主張に変換する。
        # claims = post_process_functions.claim_extraction(post=user_post)
        claims = post_process_functions.convert_post_to_claim(post=user_post)
        status = post_process_functions.status
        if not check_status(status):
            ERROR_STATUS = status
            return JSONResponse(content=ERROR_STATUS["message"], status_code=ERROR_STATUS["status"])

        # 主張を質問の形に変換する
        queries = post_process_functions.query_generation(claim_list=claims)
        status = post_process_functions.status
        if not check_status(status):
            ERROR_STATUS = status
            return JSONResponse(content=ERROR_STATUS["message"], status_code=ERROR_STATUS["status"])

        # エージェントの判断結果をもらって、戻します。
        claims_and_results_news = news_agent.news_agent_workflow(claim_list=claims)
        status = news_agent.status
        if not check_status(status):
            ERROR_STATUS = status
            return JSONResponse(content=ERROR_STATUS["message"], status_code=ERROR_STATUS["status"])

        # エージェントの判断結果をbooleanからstrに変形する
        agent_response_factuality = convert_response_from_single_agent(
            agent_response_factuality=claims_and_results_news[0]["factuality"]
        )

        return JSONResponse(content=agent_response_factuality, status_code=200)
    except Exception as err:

        traceback.print_exc()

        raise HTTPException(status_code=ERROR_STATUS["status"], detail=ERROR_STATUS["message"])


@router.post("/only_language_model")
async def language_model_fact_checking(language_model_fact_checking_payload: MultiAgentFactCheckJson):
    """
    言語モデルだけで事実検証を行う

    :param language_model_fact_checking_payload:
    :return:
    """
    global ERROR_STATUS
    try:
        # 転送された投稿内容の読み込み
        user_post = language_model_fact_checking_payload.raw

        # 投稿処理クラスの初期化
        post_process_functions = post_process.PostProcess()
        status = post_process_functions.status
        if not check_status(status):
            ERROR_STATUS = status
            return JSONResponse(content=ERROR_STATUS["message"], status_code=ERROR_STATUS["status"])

        # 言語モデルエージェントの初期化
        llm_agent = language_model_agent.LanguageModelAgent()
        status = llm_agent.status
        if not check_status(status):
            ERROR_STATUS = status
            return JSONResponse(content=ERROR_STATUS["message"], status_code=ERROR_STATUS["status"])

        # 主張の変換
        claims = post_process_functions.convert_post_to_claim(post=user_post)
        status = post_process_functions.status
        if not check_status(status):
            ERROR_STATUS = status
            return JSONResponse(content=ERROR_STATUS["message"], status_code=ERROR_STATUS["status"])

        # 主張を問題の形に変換する
        queries = post_process_functions.query_generation(claim_list=claims)
        status = post_process_functions.status
        if not check_status(status=status):
            ERROR_STATUS = status
            return JSONResponse(content=status["message"], status_code=status["status"])

        # エージェントを用いて事実検証を行う
        verification_results = llm_agent.llm_agent_workflow(claim_list=claims, query_list=queries)
        status = llm_agent.status
        if not check_status(status):
            ERROR_STATUS = status
            return JSONResponse(content=ERROR_STATUS["message"], status_code=ERROR_STATUS["status"])

        return JSONResponse(content=verification_results[0]["factuality"], status_code=200)
    except Exception as err:

        traceback.print_exc()

        raise HTTPException(status_code=ERROR_STATUS["status"], detail=ERROR_STATUS["message"])


@router.post("/multi_agent_only_language_model")
async def language_model_fact_checking(language_model_fact_checking_payload: MultiAgentFactCheckJson):
    """
    マルチ言語モデルによるエージェントを用いて事実検証を行う
    言語モデルのパラメータはtemperature = 1.0と設定する。

    :param language_model_fact_checking_payload:
    :return:
    """
    global ERROR_STATUS
    try:
        # 転送された投稿内容の読み込み
        user_post = language_model_fact_checking_payload.raw

        # 投稿処理クラスの初期化
        post_process_functions = post_process.PostProcess()
        status = post_process_functions.status
        if not check_status(status):
            ERROR_STATUS = status
            return JSONResponse(content=ERROR_STATUS["message"], status_code=ERROR_STATUS["status"])

        # 言語モデルエージェントの初期化
        llm_agent1 = language_model_agent.LanguageModelAgent()
        llm_agent1.openai.gpt_config["temperature"] = 1.0
        status = llm_agent1.status
        if not check_status(status):
            ERROR_STATUS = status
            return JSONResponse(content=ERROR_STATUS["message"], status_code=ERROR_STATUS["status"])

        llm_agent2 = language_model_agent.LanguageModelAgent()
        llm_agent2.openai.gpt_config["temperature"] = 1.0
        status = llm_agent2.status
        status = llm_agent2.status
        if not check_status(status):
            ERROR_STATUS = status
            return JSONResponse(content=ERROR_STATUS["message"], status_code=ERROR_STATUS["status"])

        llm_agent3 = language_model_agent.LanguageModelAgent()
        llm_agent3.openai.gpt_config["temperature"] = 1.0
        status = llm_agent3.status
        status = llm_agent3.status
        if not check_status(status):
            ERROR_STATUS = status
            return JSONResponse(content=ERROR_STATUS["message"], status_code=ERROR_STATUS["status"])

        # 主張の変換
        claims = post_process_functions.convert_post_to_claim(post=user_post)
        status = post_process_functions.status
        if not check_status(status):
            ERROR_STATUS = status
            return JSONResponse(content=ERROR_STATUS["message"], status_code=ERROR_STATUS["status"])

        # 主張を問題の形に変換する
        queries = post_process_functions.query_generation(claim_list=claims)
        status = post_process_functions.status
        if not check_status(status=status):
            ERROR_STATUS = status
            return JSONResponse(content=status["message"], status_code=status["status"])

        # エージェントを用いて事実検証を行う
        verification_results_1 = llm_agent1.llm_agent_workflow(claim_list=claims, query_list=queries)
        status = llm_agent1.status
        if not check_status(status):
            ERROR_STATUS = status
            return JSONResponse(content=ERROR_STATUS["message"], status_code=ERROR_STATUS["status"])

        verification_results_2 = llm_agent2.llm_agent_workflow(claim_list=claims, query_list=queries)
        status = llm_agent2.status
        if not check_status(status):
            ERROR_STATUS = status
            return JSONResponse(content=ERROR_STATUS["message"], status_code=ERROR_STATUS["status"])

        verification_results_3 = llm_agent3.llm_agent_workflow(claim_list=claims, query_list=queries)
        status = llm_agent3.status
        if not check_status(status):
            ERROR_STATUS = status
            return JSONResponse(content=ERROR_STATUS["message"], status_code=ERROR_STATUS["status"])

        # 各エージェントからもらった結果の確信度を計算する
        agent_results_list = [verification_results_1, verification_results_2, verification_results_3]
        claims_and_sub_confidence = post_process_functions.calculate_sub_confidence(
            agent_results_list=agent_results_list)

        # 主張の確信度を合わせて、投稿の信憑性（しんぴょうせい）を計算する
        post_and_confidence = post_process_functions.judge_original_post_factuality(
            claims_and_sub_confidence=claims_and_sub_confidence, original_post=user_post
        )
        print("MAFC Score:", post_and_confidence["confidence"])

        return JSONResponse(content=post_and_confidence["factuality"], status_code=200)
    except Exception as err:

        traceback.print_exc()

        raise HTTPException(status_code=ERROR_STATUS["status"], detail=ERROR_STATUS["message"])


@router.post("/fact_checking_binary")
async def multi_agent_fact_checking_binary(multi_agent_fact_check_binary_payload: MultiAgentFactCheckJson):
    """
        マルチエージェント事実検証における議論エージェントの開発はまだ決めていないですが、それに想定した上で、マルチエージェントの仕組みに
        基づくサービスがfastapiを用いて実装しました。

        :param multi_agent_fact_check_binary_payload:
        :return:
        """
    global ERROR_STATUS
    try:
        # あるところから元の投稿を転送された
        user_post = multi_agent_fact_check_binary_payload.raw

        # 投稿処理クラスの初期化
        post_process_functions = post_process.PostProcess()
        status = post_process_functions.status
        if not check_status(status=status):
            ERROR_STATUS = status
            return JSONResponse(content=status["message"], status_code=status["status"])

        # 各事実検証エージェントの初期化
        google_search_agent = google_search_fact_check_agent.GoogleSearchFactCheckAgent()
        status = google_search_agent.status
        if not check_status(status=status):
            ERROR_STATUS = status
            return JSONResponse(content=status["message"], status_code=status["status"])

        wikipedia_agent = wikipedia_fact_check_agent.WikipediaFactCheckAgent()
        status = wikipedia_agent.status
        if not check_status(status=status):
            ERROR_STATUS = status
            return JSONResponse(content=status["message"], status_code=status["status"])

        # news_agent = news_fact_check_agent.NewsFactCheckAgent()
        # status = news_agent.status
        # if not check_status(status):
        #     ERROR_STATUS = status
        #     return JSONResponse(content=status["message"], status_code=status["status"])
        llm_agent = language_model_agent.LanguageModelAgent()
        status = llm_agent.status
        if not check_status(status):
            ERROR_STATUS = status
            return JSONResponse(content=status["message"], status_code=status["status"])

        # 投稿から主張を抽出する・実験を行うときには下のメソッドを使って、そのまま主張を特定の形に変形する
        # claims = post_process_functions.claim_extraction(post=user_post)
        claims = post_process_functions.convert_post_to_claim(post=user_post)
        status = post_process_functions.status
        if not check_status(status=status):
            ERROR_STATUS = status
            return JSONResponse(content=status["message"], status_code=status["status"])

        # 主張を問題の形に変換する
        queries = post_process_functions.query_generation(claim_list=claims)
        status = post_process_functions.status
        if not check_status(status=status):
            ERROR_STATUS = status
            return JSONResponse(content=status["message"], status_code=status["status"])

        # fact checking agentの起動。主張、各エージェントの判断結果およびそれぞれの確信度を返信する
        claims_and_results_google = google_search_agent.google_search_agent_workflow(query_list=queries,
                                                                                     claim_list=claims)
        status = google_search_agent.status
        if not check_status(status=status):
            ERROR_STATUS = status
            return JSONResponse(content=status["message"], status_code=status["status"])

        claims_and_results_wiki = wikipedia_agent.wikipedia_agent_workflow(query_list=queries,
                                                                           claim_list=claims)
        status = wikipedia_agent.status
        if not check_status(status=status):
            ERROR_STATUS = status
            return JSONResponse(content=status["message"], status_code=status["status"])

        # claims_and_results_news = news_agent.news_agent_workflow(claim_list=claims)
        # status = news_agent.status
        # if not check_status(status):
        #     ERROR_STATUS = status
        #     return JSONResponse(content=status["message"], status_code=status["status"])
        claims_and_results_llm = llm_agent.llm_agent_workflow(claim_list=claims, query_list=queries)
        status = llm_agent.status
        if not check_status(status):
            ERROR_STATUS = status
            return JSONResponse(content=status["message"], status_code=status["status"])

        # 各エージェントからもらった結果の確信度を計算する
        agent_results_list = [claims_and_results_google, claims_and_results_wiki, claims_and_results_llm]
        claims_and_sub_confidence = post_process_functions.calculate_sub_confidence(
            agent_results_list=agent_results_list)

        # 主張の確信度を合わせて、投稿の信憑性（しんぴょうせい）を計算する
        post_and_confidence = post_process_functions.judge_original_post_factuality_binary(
            claims_and_sub_confidence=claims_and_sub_confidence, original_post=user_post
        )
        print("MAFC Score:", post_and_confidence["confidence"])

        return JSONResponse(content=post_and_confidence["factuality"], status_code=200)
    except Exception as err:

        traceback.print_exc()

        raise HTTPException(status_code=ERROR_STATUS["status"], detail=ERROR_STATUS["message"])


# マルチエージェントファクトチェック
@router.post("/fact_checking_extra_single_agent")
async def multi_agent_fact_checking(multi_agent_fact_check_payload: MultiAgentFactCheckJson):
    """
    マルチエージェント事実検証における議論エージェントの開発はまだ決めていないですが、それに想定した上で、マルチエージェントの仕組みに
    基づくサービスがfastapiを用いて実装しました。

    :param multi_agent_fact_check_payload:
    :return:
    """
    global ERROR_STATUS
    try:
        # あるところから元の投稿を転送された
        user_post = multi_agent_fact_check_payload.raw

        # 投稿処理クラスの初期化
        post_process_functions = post_process.PostProcess()
        status = post_process_functions.status
        if not check_status(status=status):
            ERROR_STATUS = status
            return JSONResponse(content=status["message"], status_code=status["status"])

        # 各事実検証エージェントの初期化
        google_search_agent = google_search_fact_check_agent.GoogleSearchFactCheckAgent()
        status = google_search_agent.status
        if not check_status(status=status):
            ERROR_STATUS = status
            return JSONResponse(content=status["message"], status_code=status["status"])

        wikipedia_agent = wikipedia_fact_check_agent.WikipediaFactCheckAgent()
        status = wikipedia_agent.status
        if not check_status(status=status):
            ERROR_STATUS = status
            return JSONResponse(content=status["message"], status_code=status["status"])

        # news_agent = news_fact_check_agent.NewsFactCheckAgent()
        # status = news_agent.status
        # if not check_status(status):
        #     ERROR_STATUS = status
        #     return JSONResponse(content=status["message"], status_code=status["status"])
        llm_agent = language_model_agent.LanguageModelAgent()
        status = llm_agent.status
        if not check_status(status):
            ERROR_STATUS = status
            return JSONResponse(content=status["message"], status_code=status["status"])

        # 投稿から主張を抽出する・実験を行うときには下のメソッドを使って、そのまま主張を特定の形に変形する
        # claims = post_process_functions.claim_extraction(post=user_post)
        claims = post_process_functions.convert_post_to_claim(post=user_post)
        status = post_process_functions.status
        if not check_status(status=status):
            ERROR_STATUS = status
            return JSONResponse(content=status["message"], status_code=status["status"])

        # 主張を問題の形に変換する
        queries = post_process_functions.query_generation(claim_list=claims)
        status = post_process_functions.status
        if not check_status(status=status):
            ERROR_STATUS = status
            return JSONResponse(content=status["message"], status_code=status["status"])

        # fact checking agentの起動。主張、各エージェントの判断結果およびそれぞれの確信度を返信する
        claims_and_results_google = google_search_agent.google_search_agent_workflow(query_list=queries,
                                                                                     claim_list=claims)
        status = google_search_agent.status
        if not check_status(status=status):
            ERROR_STATUS = status
            return JSONResponse(content=status["message"], status_code=status["status"])

        claims_and_results_wiki = wikipedia_agent.wikipedia_agent_workflow(query_list=queries,
                                                                           claim_list=claims)
        status = wikipedia_agent.status
        if not check_status(status=status):
            ERROR_STATUS = status
            return JSONResponse(content=status["message"], status_code=status["status"])

        # claims_and_results_news = news_agent.news_agent_workflow(claim_list=claims)
        # status = news_agent.status
        # if not check_status(status):
        #     ERROR_STATUS = status
        #     return JSONResponse(content=status["message"], status_code=status["status"])
        claims_and_results_llm = llm_agent.llm_agent_workflow(claim_list=claims, query_list=queries)
        status = llm_agent.status
        if not check_status(status):
            ERROR_STATUS = status
            return JSONResponse(content=status["message"], status_code=status["status"])

        # 各エージェントからもらった結果の確信度を計算する
        agent_results_list = [claims_and_results_google, claims_and_results_wiki, claims_and_results_llm]
        claims_and_sub_confidence = post_process_functions.calculate_sub_confidence(
            agent_results_list=agent_results_list)

        # 主張の確信度を合わせて、投稿の信憑性（しんぴょうせい）を計算する
        post_and_confidence = post_process_functions.judge_original_post_factuality(
            claims_and_sub_confidence=claims_and_sub_confidence, original_post=user_post
        )
        print("MAFC Score:", post_and_confidence["confidence"])

        return JSONResponse(content=post_and_confidence["factuality"], status_code=200)
    except Exception as err:

        traceback.print_exc()

        raise HTTPException(status_code=ERROR_STATUS["status"], detail=ERROR_STATUS["message"])
