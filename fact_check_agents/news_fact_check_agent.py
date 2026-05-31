__author__ = "Dong Yihan"

import os
'''
主張からキーワードを抽出し、newsapiを用いて検索する
'''

from keybert import KeyBERT
from pynytimes import NYTAPI

from dependencies import wikipedia_client
from .base_agent import BaseAgent


class NewsFactCheckAgent(BaseAgent):

    def __init__(self):
        super().__init__(agent_name="news_fact_check_agent", agent_weight=1.0)

        self.news_api_key = os.getenv("NYTIMES_API_KEY", "")
        self.news_client = NYTAPI(key=self.news_api_key, parse_dates=True) if self.news_api_key else None
        self.keywords_extraction_instruction = """
        I have the following document:
        [DOCUMENT]

        Based on the information above, extract two keywords that best describe the objects of the text.
        Make sure to only extract keywords that appear in the text.
        Use the following format separated by commas:
        <keywords>
        """
        self.keyword_model = KeyBERT(model=self.embedding_model)

        self.status.update(status=200, message="successfully initialize news agent")
        return

    def keywords_extraction(self, claim_list):
        """
        主張からキーワードを抽出するメソッド。

        :param claim_list: list[dict{"claim": "claim}] 主張が保存されているリスト
        :return: キーワードリスト
        """
        self.status.update(status=500, message="error in keywords_extraction")

        # list[dict]からlistに変形する
        claims = [claim["claim"] for claim in claim_list]
        keywords_list = self.keyword_model.extract_keywords(
            docs=claims,
            keyphrase_ngram_range=(1, 2),
            stop_words="english",
            top_n=2
        )
        keywords = [self._format_keywords(keyword_candidates, claim) for keyword_candidates, claim in zip(keywords_list, claims)]

        self.status.update(status=200, message="successfully extract keywords from claims")
        return claims, keywords

    def search_news(self, claims: list[str], keywords: list[str]):
        """
        キーワードを検索し、ニュースを獲得するメソッド

        ここで検索したニュースの中に一番近い結果を選ぶ。検索されたニュースと主張のBERTScoreを計算し、関係性を測ります。
        :param claims:　主張が保存されているリスト　list[str]
        :param keywords:　ニュースを検索するため、抽出されたキーワードリスト　list[str]
        :return: 主張、ニュース及びBERTScoreの組み合わせ　list[dict{"claim": claim, "evidence": news, "bert_score": BERTScore}]
        """
        self.status.update(status=500, message="error in search_news")

        claims_and_news = []
        if len(claims) != len(keywords):
            self.status.update(status=500, message="check program! the length of claims does not equal to the length"
                                                   " of keywords")
            return claims_and_news

        # 抽出されたキーワードを用いて、ニュースを検索する。主張、検索されたニュースと信憑性を組み合わせて保存する。
        for i in range(0, len(keywords)):

            if self.news_client:
                # resultsはlist[dict{"abstract", "snippet"}]です
                # 全てのまとめを抽出する
                results = self.news_client.article_search(query=keywords[i], options={"sort": "relevance"}, results=5)
                if results:
                    snippets = results[0]["snippet"]
                else:
                    snippets = ""
            else:
                snippets = self.search_wikipedia_fallback(query=keywords[i])

            # 主張と検索されたニュースのBERTScoreを計算する
            # bert_score_f1 = self.calculate_bert_score(claim=claims[i], evidence=snippets)

            claim_and_news = {
                "claim": claims[i],
                "evidence": snippets
                # "bert_score": bert_score_f1
            }
            claims_and_news.append(claim_and_news)

        self.status.update(status=200, message="successfully search news")
        return claims_and_news

    @staticmethod
    def _format_keywords(keyword_candidates, fallback_claim):
        keywords = []
        for candidate in keyword_candidates:
            if isinstance(candidate, tuple) and candidate:
                keywords.append(candidate[0])
            elif isinstance(candidate, str):
                keywords.append(candidate)
        return " ".join(keywords[:2]) if keywords else fallback_claim

    @staticmethod
    def search_wikipedia_fallback(query: str):
        for title in wikipedia_client.search_titles(query=query, limit=3):
            summary = wikipedia_client.page_summary(title=title, sentences=2)
            if summary:
                return summary
        return ""

    def claims_verification(self, claims_and_news=None, relevance_list=None):
        """
        主張の正しさを判断するメソッド

        :param relevance_list: list[str] 主張と情報の関連度が保存されたリスト
        :param claims_and_news: list[dict{"claim": claim, "evidence": news, "bert_score": BERTScore}]
        :return: 言語モデルに基づく判断結果です。
        """
        if claims_and_news is None:
            claims_and_news = []
        self.status.update(status=500, message="error in claims_verification of the agent {}".format(self.agent_name))

        verification_results = []
        for i in range(0, len(claims_and_news)):
            claim_and_news = claims_and_news[i]
            relevance = relevance_list[i]
            verification_prompts = [
                {
                    "role": "system",
                    "content": self.claim_verification_instruct["system"]
                },
                {
                    "role": "user",
                    "content": self.claim_verification_instruct["user"].format(
                        claim=claim_and_news["claim"],
                        evidence=claim_and_news["evidence"],
                        # relevance=relevance
                    )
                }
            ]

            verification_result = self.openai.get_gpt_response(prompts=verification_prompts)

            # 具体的な原因は分かりませんが、gptからの返信の形が違いようです。
            verification_result = verification_result.replace("true", "TRUE")
            verification_result = verification_result.replace("false", "FALSE")
            verification_result = verification_result.replace("TRUE", "True")
            verification_result = verification_result.replace("FALSE", "False")

            # 返信の中のnullを削除する
            verification_result = verification_result.replace("null", "None")
            print("news_verification_result:", verification_result)
            result_dict = eval(verification_result)
            result_dict["confidence"] = float(result_dict["confidence"])

            # 確信度スコアを計算する
            # result_dict["score"] = claim_and_news["bert_score"]
            # verification_results.append(result_dict)
            # 言語モデルから判断の確信度を取得して、dictのキーの名前を変える
            result_dict["score"] = result_dict["confidence"]
            del result_dict["confidence"]
            result_dict["claim"] = claim_and_news["claim"]
            verification_results.append(result_dict)

        self.status.update(status=200, message="successfully judge if the claims are correct or not")
        return verification_results

    def news_agent_workflow(self, claim_list):
        """
        news agentの流れをまとめる。

        各エージェントのinputとoutputを同一化するため、各エージェントに別々に流れをまとめるメソッドの作成が必要です。
        :param claim_list: list[dict{"claim": "claim}]　主張が保存されたリスト
        :return: ニュースエージェントの判断結果です
        """
        self.status.update(status=500, message="error in news_agent_workflow")

        # 主張とそれぞれのキーワードを取得する
        claims, keywords = self.keywords_extraction(claim_list=claim_list)

        # ニュースを検索する
        claims_and_news = self.search_news(claims=claims, keywords=keywords)

        # 情報と主張の関連度を値踏みする
        relevance_list = self.evaluate_relevance(claim_evidence_list=claims_and_news)

        # 主張の正しさを判断する
        verification_results = self.claims_verification(claims_and_news=claims_and_news, relevance_list=relevance_list)

        # このエージェントにより主張の確信度とこのエージェントのウェイトをかけて、再計算する →　リファクタリング必要あり
        # 今verification_results["score"]は主張により変わりますので、かける必要がなさそうです
        claims_and_results = self.combine_confidence_with_claims(verification_results=verification_results,
                                                                 claims=claim_list)
        print("result of the {} agent: {}".format(self.agent_name, claims_and_results))

        self.status.update(status=200, message="successfully fact check with {}".format(self.agent_name))
        return claims_and_results
