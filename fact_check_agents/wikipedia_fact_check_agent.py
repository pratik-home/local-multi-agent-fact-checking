__author__ = "Dong Yihan"

import requests
import json
from keybert import KeyBERT
from dependencies import wikipedia_client
from .base_agent import BaseAgent


# wikipediaの内容のみを用いて事実検証を行うエージェント
class WikipediaFactCheckAgent(BaseAgent):

    def __init__(self):
        super().__init__(agent_name="wikipedia_fact_check_agent", agent_weight=1.0)

        self.keywords_extraction_instruction = """
                I have the following document:
                [DOCUMENT]

                Based on the information above, extract two keywords that best describe the objects of the text.
                Make sure to only extract keywords that appear in the text.
                Use the following format separated by commas:
                <keywords>
                """
        self.keyword_model = KeyBERT(model=self.embedding_model)

        self.status.update(status=200, message="successfully initialize wikipedia agent")
        return

    def keywords_extraction(self, queries: list[str]):
        """
        質問からキーワードを抽出するメソッド。

        :param queries: list[str] 質問が保存されているリスト
        :return: キーワードリスト
        """
        self.status.update(status=500, message="error in keywords_extraction")

        keywords_list = self.keyword_model.extract_keywords(
            docs=queries,
            keyphrase_ngram_range=(1, 2),
            stop_words="english",
            top_n=2
        )
        keywords = [self._format_keywords(keyword_candidates, query) for keyword_candidates, query in zip(keywords_list, queries)]

        self.status.update(status=200, message="successfully extract keywords from claims")
        return keywords

    def search_wikipedia_titles(self, queries: list[str]):
        """
        queryからwikipediaの検索を行い、wikiページのタイトルを取得する。一つのqueryに対し、最適なページだけを取得する。
        :param queries: list[str]
        :return:
        """
        self.status.update(status=500, message="error in search_wikipedia_titles")

        wiki_titles = []  # list[str]

        wiki_keywords = self.keywords_extraction(queries=queries)
        for wiki_keyword in wiki_keywords:
            wiki_pages = wikipedia_client.search_titles(query=wiki_keyword, limit=5)
            if wiki_pages:
                wiki_title = wiki_pages[0]
            else:
                wiki_title = ""
            wiki_titles.append(wiki_title)

        self.status.update(status=200, message="successfully achieve titles of wikipedia related to queries")
        return wiki_titles  # list[str]

    @staticmethod
    def _format_keywords(keyword_candidates, fallback_query):
        """
        KeyBERT returns [(keyword, score), ...] for one document. Keep the
        Wikipedia search query compact, and fall back to the original query
        when no keywords are available.
        """
        keywords = []
        for candidate in keyword_candidates:
            if isinstance(candidate, tuple) and candidate:
                keywords.append(candidate[0])
            elif isinstance(candidate, str):
                keywords.append(candidate)
        return " ".join(keywords[:2]) if keywords else fallback_query

    def search_wikipedia_pages(self, title_list: list[str]):
        """
        wikipediaからタイトルを用いて文章の本文を取得する
        :param title_list: list[str] タイトルのリスト
        :return: wiki_pages: list[str] 関連ページの情報です
        """
        self.status.update(status=500, message="error in search_wikipedia_pages")

        wiki_pages = []
        for title in title_list:
            if title:
                wiki_summary = wikipedia_client.page_summary(title=title, sentences=5)
            else:
                wiki_summary = ""
            wiki_pages.append(wiki_summary)

        self.status.update(status=200, message="successfully achieve contents on wikipedia pages")
        return wiki_pages

    def search_wikipedia_queries(self, query_list, claim_list):
        """
        wikipediaの検索機能を用いて関連ページの検索
        :param query_list: list[dict{"query": "query", "answer": "answer"}]質問のリスト
        :param claim_list: [dict{"claim": "claim}]主張のリスト
        :return: claims_and_wiki_pages: list[dict{"claim": "claim1", "evidence": "wiki_page1"}]
        主張と検索された関連ページの結果です
        """
        self.status.update(status=500, message="error in search_wikipedia_queries")

        # 主張とwikipediaから得た内容を合わせる
        claims_and_wiki_pages = []
        if len(query_list) != len(claim_list):
            self.status.update(status=500, message="the length of the query list does not"
                                                   " equal to that of the claim list")
            print("check program! the claim list and the query list should have the same length")
        else:
            # list[dict]からlistに変形する
            queries = [query["query"] for query in query_list]
            wiki_titles = self.search_wikipedia_titles(queries=queries)
            wiki_pages = self.search_wikipedia_pages(title_list=wiki_titles)
            # query_list claim_listとwiki_pagesの長さは同じはずだ
            for i in range(0, len(wiki_pages)):
                claim_and_wiki_page = {
                    "claim": claim_list[i]["claim"],
                    "evidence": wiki_pages[i]
                }
                claims_and_wiki_pages.append(claim_and_wiki_page)
                print("wiki_evidence:\n", wiki_pages[i])
            self.status.update(status=200, message="successfully get claims and contents from wiki pages")

        return claims_and_wiki_pages

    def claims_verification(self, claims_and_wiki_pages=None, relevance_list=None):
        """
        主張の正しさを判断するメソッド
        :param relevance_list: list[str] 主張と情報の関連度が保存されたリスト
        :param claims_and_wiki_pages: list[dict{"claim": "claim1", "evidence": "wiki_page1"}]
        主張とwikipediaの関連ページを合わせて保存されたリスト。search_wikipedia_queries()から得た結果です。
        :return: 言語モデルに基づく判断結果です
        """
        if claims_and_wiki_pages is None:
            claims_and_wiki_pages = []
        self.status.update(status=500, message="error in claims_verification of wikipedia agent")

        verification_results = []
        for i in range(0, len(claims_and_wiki_pages)):
            claim_and_wiki_page = claims_and_wiki_pages[i]
            relevance = relevance_list[i]
            verification_prompts = [
                {
                    "role": "system",
                    "content": self.claim_verification_instruct["system"]
                },
                {
                    "role": "user",
                    "content": self.claim_verification_instruct["user"].format(
                        claim=claim_and_wiki_page["claim"],
                        evidence=claim_and_wiki_page["evidence"],
                        relevance=relevance
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
            print("wikipedia_verification_result:", verification_result)
            result_dict = eval(verification_result)
            result_dict["confidence"] = float(result_dict["confidence"])

            # 言語モデルにより判断結果と結果の確信度を獲得する。
            result_dict["score"] = result_dict["confidence"]
            del result_dict["confidence"]
            verification_results.append(result_dict)

        self.status.update(status=200, message="successfully judge if the claims are correct or not")
        return verification_results

    def wikipedia_agent_workflow(self, query_list, claim_list):
        """
        wikipedia agentの流れをまとめる。

        各エージェントのinputとoutputを同一化するため、各エージェントに別々に流れをまとめるメソッドの作成が必要です。
        ここのquery_listとclaim_listはpost_processのメソッドで投稿から抽出したもの
        :param query_list: list[dict{"query": "query", "answer": "answer"}]　質問が保存されたリスト
        :param claim_list: list[dict{"claim": "claim}]　主張が保存されたリスト
        :return: wikipediaエージェントが判断した結果です
        """
        self.status.update(status=500, message="error in wikipedia_agent_workflow")

        # 主張と検索されたwiki pageを合わせる
        claims_and_wiki_pages = self.search_wikipedia_queries(query_list=query_list, claim_list=claim_list)

        # 情報と主張の関連度を値踏みする
        relevance_list = self.evaluate_relevance(claim_evidence_list=claims_and_wiki_pages)

        # 主張は正しいかどうかを判断する
        verification_results = self.claims_verification(claims_and_wiki_pages=claims_and_wiki_pages,
                                                        relevance_list=relevance_list)

        # このエージェントにより主張の確信度とこのエージェントのウェイトをかけて、再計算する
        claims_and_results = self.combine_confidence_with_claims(verification_results=verification_results,
                                                                 claims=claim_list)
        print("result of the {} agent: {}".format(self.agent_name, claims_and_results))

        self.status.update(status=200, message="successfully check fact using {}".format(self.agent_name))
        return claims_and_results
