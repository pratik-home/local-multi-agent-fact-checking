__author__ = "Dong Yihan"

import requests
import json
from .base_agent import BaseAgent


class LanguageModelAgent(BaseAgent):
    """
    比較実験を行うため、言語モデルだけで主張の信ぴょう性を判断するエージェントの作成

    """

    def __init__(self):
        """
        言語モデルエージェントの初期化。evidenceが不要なので、他のプロンプトを導入する
        """
        super().__init__(agent_name="language_model_agent", agent_weight=1.0)

        with open("./prompts/agreement_verification_with_relevance.json", "r") as claim_verification_file:
            verification_data = json.load(claim_verification_file)
        self.claim_verification_instruct = {
            "system": verification_data["system"],
            "user": verification_data["user"]
        }

        with open("./prompts/llm_generate_answer.json", "r") as llm_answer_generation_file:
            llm_answer_data = json.load(llm_answer_generation_file)
        self.llm_answer_generation_instruct = {
            "system": llm_answer_data["system"],
            "user": llm_answer_data["user"]
        }

        self.status.update(status=200, message="successfully initialize language model agent")
        return

    def claims_verification(self, claims_and_evidence_list=None, relevance_list=None):
        """

        :param relevance_list: list[str] 主張と情報の関連度が保存されたリスト
        :param claims_and_evidence_list: list[dict{"claim": claim, "evidence: evidence}] 主張と情報が保存されたリスト。
        :return: verification_results: list[dict{"claim", "factuality", "confidence"}]
        """

        if claims_and_evidence_list is None:
            verification_results = []
            self.status.update(status=500, message="error in claims_verification of llm agent")
            return verification_results

        verification_results = []
        for i in range(0, len(claims_and_evidence_list)):
            element = claims_and_evidence_list[i]
            relevance = relevance_list[i]
            verification_prompts = [
                {
                    "role": "system",
                    "content": self.claim_verification_instruct["system"]
                },
                {
                    "role": "user",
                    "content": self.claim_verification_instruct["user"].format(
                        claim=element["claim"], evidence=element["evidence"], relevance=relevance
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
            print("language_model_verification_result:", verification_result)
            print("type of verification result of llm", type(verification_result))
            result_dict = eval(verification_result)

            # 言語モデルにより判断結果と結果の確信度を獲得する。
            result_dict["score"] = result_dict["confidence"]
            del result_dict["confidence"]
            result_dict["claim"] = element["claim"]
            verification_results.append(result_dict)

        self.status.update(status=200, message="successfully judge if the claims are correct or not")
        return verification_results

    def generate_answer_to_queries(self, query_list=None, claim_list=None):
        """
        言語モデルは各問題に対し答えを生成するメソッド

        :param claim_list: list[dict{"claim": claim}]
        :param query_list: list[dict{"query": "query", "answer": "answer"}] 質問が保存されたリストです。
        :return: llm_claims_answers: list[dict{"claim": claim, "evidence: evidence}]
        """
        if query_list is None:
            llm_answers = []
            self.status.update(status=500, message="error in claims_verification of google search agent")
            return llm_answers

        llm_claims_answers = []
        for i in range(0, len(query_list)):
            query = query_list[i]
            claim = claim_list[i]
            answer_generation_prompt = [
                {
                    "role": "system",
                    "content": self.llm_answer_generation_instruct["system"]
                },
                {
                    "role": "user",
                    "content": self.llm_answer_generation_instruct["user"].format(
                        query=query["query"]
                    )
                }
            ]

            llm_answer = self.openai.get_gpt_response(prompts=answer_generation_prompt)

            print("language_model_answer:", llm_answer)
            claim_and_answer = {
                "claim": claim["claim"],
                "evidence": llm_answer
            }
            llm_claims_answers.append(claim_and_answer)

        self.status.update(status=200, message="successfully generate llm answers to queries")
        return llm_claims_answers

    def llm_agent_workflow(self, claim_list, query_list):
        """
        google search agentの流れをまとめる。
        各エージェントのinputとoutputを同一化するため、各エージェントに別々に流れをまとめるメソッドの作成が必要です。
        ここのclaim_listはpost_processのメソッドで投稿から抽出したもの。
        :param query_list: query_list: list[dict{"query": "query", "answer": "answer"}] 質問が保存されたリストです。
        :param claim_list: list[dict{"claim": "claim}] 主張が保存されたリストです
        :return:　Googleエージェントが判断した結果です
        """
        self.status.update(status=500, message="error in google_search_agent_workflow")

        # 言語モデルで主張に関する情報を生成する
        llm_claims_and_evidence = self.generate_answer_to_queries(query_list=query_list, claim_list=claim_list)

        # 情報と主張の関連度を値踏みする
        relevance_list = self.evaluate_relevance(claim_evidence_list=llm_claims_and_evidence)

        # 主張は正しいかどうかを判断する
        claim_verification_results = self.claims_verification(claims_and_evidence_list=llm_claims_and_evidence,
                                                              relevance_list=relevance_list)

        # 情報源の信憑性によりエージェントの判断結果の確信度を計算し、各エージェントの判断結果を同一化する。
        claims_and_results = self.combine_confidence_with_claims(verification_results=claim_verification_results,
                                                                 claims=claim_list)
        print("result of the {}: {}".format(self.agent_name, claims_and_results))

        self.status.update(status=200, message="successfully check fact using {}".format(self.agent_name))
        return claims_and_results
