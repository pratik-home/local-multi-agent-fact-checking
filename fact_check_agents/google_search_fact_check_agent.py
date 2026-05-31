__author__ = "Dong Yihan"

import os
import requests
import json
from bs4 import BeautifulSoup
from urllib.parse import parse_qs, urlparse
from .base_agent import BaseAgent


class GoogleSearchFactCheckAgent(BaseAgent):
    """
    google search api を用いて事実検証を行うエージェント
    """

    def __init__(self):
        super().__init__(agent_name="web_search_fact_check_agent", agent_weight=1.0)

        # Optional Serper key. Without it, this agent falls back to DuckDuckGo.
        self.serper_key = os.getenv("SERPER_API_KEY", "")

        self.status.update(status=200, message="successfully initialize google search agent")
        return

    def get_google_responses(self, queries: list[str]):
        """
        google search apiを用いて検索するメソッド。
        :param queries: list[str] 検索したい問題のリストです
        :return: Googleで検索した結果　list[dict]
        """
        self.status.update(status=500, message="error in get_google_responses")

        if not self.serper_key:
            google_response = [self.get_duckduckgo_fallback_response(query=query) for query in queries]
            self.status.update(status=200, message="successfully get responses from duckduckgo fallback")
            return google_response

        google_search_url = "https://google.serper.dev/search"
        queries_dict_list = []

        # strからdictに変換する
        for query in queries:
            query_dict = {"q": query, "gl": "us"}  # ここで地域も指定できますが、どんな影響を与えられるのはまだわかりません
            queries_dict_list.append(query_dict)

        payload = json.dumps(queries_dict_list)
        headers = {
            "X-API-KEY": self.serper_key,
            "Content-Type": "application/json"
        }

        google_response = requests.request(method="POST", url=google_search_url, headers=headers, data=payload)
        google_response = google_response.json()  # list[dict]

        self.status.update(status=200, message="successfully get responses from google")
        return google_response

    @staticmethod
    def get_duckduckgo_fallback_response(query: str):
        """
        Serper-free web-search fallback that returns a Serper-like response
        shape so the existing parser can keep working without a paid search API.
        """
        organic_results = []
        try:
            response = requests.get(
                "https://html.duckduckgo.com/html/",
                params={"q": query},
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=15
            )
            response.raise_for_status()
        except requests.RequestException as err:
            print("DuckDuckGo search failed:", err)
            response = None

        if response is not None:
            soup = BeautifulSoup(response.text, "html.parser")
            for result in soup.select(".result"):
                title_link = result.select_one(".result__a")
                snippet = result.select_one(".result__snippet")
                if not title_link:
                    continue

                title = title_link.get_text(" ", strip=True)
                link = GoogleSearchFactCheckAgent._normalize_duckduckgo_url(
                    title_link.get("href", "None")
                )
                snippet_text = snippet.get_text(" ", strip=True) if snippet else title
                organic_results.append({
                    "title": title,
                    "snippet": snippet_text,
                    "link": link
                })
                if len(organic_results) >= 5:
                    break

        if not organic_results:
            organic_results.append({
                "title": "No DuckDuckGo result",
                "snippet": "No good DuckDuckGo fallback result was found",
                "link": "None"
            })

        return {"organic": organic_results}

    @staticmethod
    def _normalize_duckduckgo_url(url: str):
        parsed = urlparse(url)
        redirect_target = parse_qs(parsed.query).get("uddg")
        if redirect_target:
            return redirect_target[0]
        return url

    @staticmethod
    def parse_search_results(google_search_results):
        """
        googleで検索した結果をフィルタリングするメソッド
        FactToolに参照した上で作成したものですので、そこまで詳しく解読していない。でもうまく動いた
        :param google_search_results:
        :return:
        """
        snippets = []

        if google_search_results.get("answerBox"):
            answer_box = google_search_results.get("answerBox", {})
            if answer_box.get("answer"):
                element = {"content": answer_box.get("answer"), "source": "None"}
                return [element]
            elif answer_box.get("snippet"):
                element = {"content": answer_box.get("snippet").replace("\n", " "), "source": "None"}
                return [element]
            elif answer_box.get("snippetHighlighted"):
                element = {"content": answer_box.get("snippetHighlighted"), "source": "None"}
                return [element]

        if google_search_results.get("knowledgeGraph"):
            kg = google_search_results.get("knowledgeGraph", {})
            title = kg.get("title")
            entity_type = kg.get("type")
            if entity_type:
                element = {"content": f"{title}: {entity_type}", "source": "None"}
                snippets.append(element)
            description = kg.get("description")
            if description:
                element = {"content": description, "source": "None"}
                snippets.append(element)
            for attribute, value in kg.get("attributes", {}).items():
                element = {"content": f"{attribute}: {value}", "source": "None"}
                snippets.append(element)

        for result in google_search_results["organic"][:5]:
            if "snippet" in result:
                element = {"content": result["snippet"], "source": result["link"]}
                snippets.append(element)
            for attribute, value in result.get("attributes", {}).items():
                element = {"content": f"{attribute}: {value}", "source": result["link"]}
                snippets.append(element)

        if len(snippets) == 0:
            element = {"content": "No good Google Search Result was found", "source": "None"}
            return [element]

        # keep only the first 5 snippets
        snippets = snippets[:5]

        return snippets

    def search_google_queries(self, query_list, claim_list):
        """
        変換したqueryをgoogleに投げるメソッド
        :param query_list: list[dict{"query": str}]変換されたqueryが保存されたリスト。
        :param claim_list: list[dict{"claim": str}]主張が保存されたリスト。
        :return: claims_and_searching_results: list[{"claim": claim_1, "evidence": search_result1}, {}]
        主張と検索された情報を合わせて保存するリスト。
        """
        self.status.update(status=500, message="error in search_google_queries")

        # 主張と検索した結果を合わせて保存する
        claims_and_searching_results = []
        if len(query_list) != len(claim_list):
            self.status.update(status=500, message="the length of the query list does not"
                                                   " equal to that of the claim list")
            print("check program! the claim list and the query list should have the same length")
        else:
            # googleから検索結果を取得する
            queries = [query["query"] for query in query_list]
            google_responses = self.get_google_responses(queries=queries)

            #  googleから検索した結果をフィルタリングして、主張と合わせる
            for i in range(0, len(google_responses)):
                filtered_results = self.parse_search_results(google_search_results=google_responses[i])
                filtered_results_list = [output["content"] for output in filtered_results]
                filtered_result = " ".join(filtered_results_list)
                claim_and_searching_result = {
                    "claim": claim_list[i]["claim"],
                    "evidence": filtered_result
                }
                claims_and_searching_results.append(claim_and_searching_result)
                print("web_search_evidence:\n", filtered_result)
                self.status.update(status=200, message="successfully get claims and searching results of web search")

        return claims_and_searching_results

    def claims_verification(self, claims_and_searching_results=None, relevance_list=None):
        """
        google search agentに主張の正しさを判断するメソッド
        パラメーターはsearch_google_query()のreturnです
        :param relevance_list: list[str] 主張と情報の関連度が保存されたリスト
        :param claims_and_searching_results: [{"claim": "claim_1", "evidence": "search_result1"}, {}]　
        search_google_query()から得た結果です。主張と検索された情報の組み合わせ
        :return: 言語モデルに基づく判断結果です
        """
        if claims_and_searching_results is None:
            claims_and_searching_results = []
        self.status.update(status=500, message="error in claims_verification of google search agent")

        verification_results = []
        for i in range(0, len(claims_and_searching_results)):
            claim_and_searching_result = claims_and_searching_results[i]
            relevance = relevance_list[i]
            verification_prompts = [
                {
                    "role": "system",
                    "content": self.claim_verification_instruct["system"]
                },
                {
                    "role": "user",
                    "content": self.claim_verification_instruct["user"].format(
                        claim=claim_and_searching_result["claim"],
                        evidence=claim_and_searching_result["evidence"],
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
            print("web_search_verification_result:", verification_result)
            result_dict = eval(verification_result)
            result_dict["confidence"] = float(result_dict["confidence"])

            # 言語モデルにより判断結果と結果の確信度を獲得する。
            result_dict["score"] = result_dict["confidence"]
            del result_dict["confidence"]
            verification_results.append(result_dict)

        self.status.update(status=200, message="successfully judge if the claims are correct or not")
        return verification_results

    def google_search_agent_workflow(self, query_list, claim_list):
        """
        google search agentの流れをまとめる。
        各エージェントのinputとoutputを同一化するため、各エージェントに別々に流れをまとめるメソッドの作成が必要です。
        ここのquery_listとclaim_listはpost_processのメソッドで投稿から抽出したもの。
        :param query_list: list[dict{"query": "query", "answer": "answer"}] 質問が保存されたリストです。
        :param claim_list: list[dict{"claim": "claim}] 主張が保存されたリストです
        :return:　Googleエージェントが判断した結果です
        """
        self.status.update(status=500, message="error in google_search_agent_workflow")

        # queryの答えとevidenceを検索する
        claims_and_supports = self.search_google_queries(query_list=query_list, claim_list=claim_list)

        # 情報と主張の関連度を値踏みする
        relevance_list = self.evaluate_relevance(claim_evidence_list=claims_and_supports)

        # 主張は正しいかどうかを判断する
        claim_verification_results = self.claims_verification(claims_and_searching_results=claims_and_supports,
                                                              relevance_list=relevance_list)

        # 情報源の信憑性によりエージェントの判断結果の確信度を計算し、各エージェントの判断結果を同一化する。
        claims_and_results = self.combine_confidence_with_claims(verification_results=claim_verification_results,
                                                                 claims=claim_list)
        print("result of the {}: {}".format(self.agent_name, claims_and_results))

        self.status.update(status=200, message="successfully check fact using {}".format(self.agent_name))
        return claims_and_results
